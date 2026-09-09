# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
import time

import numpy as np
from torch.utils.data import Dataset

from .dataset_util import (
    crop_image_depth_and_intrinsic_by_pp,
    depth_edge_opencv,
    depth_to_world_coords_points,
    resize_image_depth_and_intrinsic,
)
from .rotation_util import rotate_90_degrees


class BaseDataset(Dataset):
    """
    Base dataset class for VGGT and VGGSfM training.

    This abstract class handles common operations like image resizing,
    augmentation, and coordinate transformations. Concrete dataset
    implementations should inherit from this class.

    Attributes:
        img_size: Target image size (typically the width)
        patch_size: Size of patches for vit
        augs.scales: Scale range for data augmentation [min, max]
        rescale: Whether to rescale images
        rescale_aug: Whether to apply augmentation during rescaling
    """

    def __init__(
        self,
        common_conf,
    ):
        """
        Initialize the base dataset with common configuration.

        Args:
            common_conf: Configuration object with the following properties, shared by all datasets:
                - img_size: Default is 512
                - patch_size: Default is 16
                - augs.scales: Default is [0.8, 1.2]
                - rescale: Default is True
                - rescale_aug: Default is True
                - random_rotation_ratio: Default is -1. If > 0, apply random rotation with this probability.
                - rescale_prob: Default is -1. If > 0, randomly apply rescale with this probability per sample.
                - avoid_upsample: Default is False. If True, never rescale when it would require upsampling.
                - skip_extreme_rescale: Default is False. If True, skip rescale when:
                    1. Any dimension needs upsampling (to avoid blocky depth map artifacts)
                    2. Downsampling ratio exceeds max_downsample_ratio (to avoid detail loss)
                - max_downsample_ratio: Default is 1.5. Maximum allowed downsampling ratio.
                    If skip_extreme_rescale=True and scale < 1/max_downsample_ratio, rescale is skipped.
        """
        super().__init__()
        self.img_size = common_conf.img_size
        self.patch_size = common_conf.patch_size
        self.aug_scale = common_conf.augs.scales
        self.rescale = common_conf.rescale
        self.rescale_aug = common_conf.rescale_aug
        self.aug_principal_point = common_conf.aug_principal_point
        self.shared_intrinsics_resize_scale = getattr(
            common_conf, "shared_intrinsics_resize_scale", False
        )
        self.random_rotation_ratio = getattr(common_conf, "random_rotation_ratio", -1)
        self.rescale_prob = getattr(common_conf, "rescale_prob", -1)
        self.avoid_upsample = getattr(common_conf, "avoid_upsample", False)
        self.same_preprocess_ratio = getattr(common_conf, "same_preprocess_ratio", -1)
        self.samples_per_weight_unit = getattr(
            common_conf, "samples_per_weight_unit", 1000
        )
        # Skip rescale when extreme scaling is needed to avoid depth map artifacts
        # - Upsampling: any scale > 1.0 causes blocky artifacts with INTER_NEAREST
        # - Downsampling: scale < 1/max_downsample_ratio causes loss of detail
        self.skip_extreme_rescale = getattr(common_conf, "skip_extreme_rescale", False)
        self.max_downsample_ratio = getattr(common_conf, "max_downsample_ratio", 1.2)
        self.fixed_base_side = getattr(common_conf, "fixed_base_side", False)
        self.shared_edge_rtol_thres = getattr(
            common_conf, "shared_edge_rtol_thres", -1.0
        )
        self.token_number = (self.img_size // self.patch_size) ** 2

        if self.img_size % self.patch_size != 0:
            raise ValueError(
                f"img_size ({self.img_size}) must be divisible by patch_size ({self.patch_size})"
            )

    def resolve_edge_rtol_thres(
        self,
        edge_rtol_thres,
        enable_edge_rtol_thres=False,
    ) -> float:
        """
        Resolve the effective depth-edge filtering threshold.

        `shared_edge_rtol_thres >= 0` globally overrides all per-dataset settings.
        Otherwise, preserve the historical behavior gated by
        `enable_edge_rtol_thres`.
        """
        shared_edge_rtol_thres = self.shared_edge_rtol_thres
        if shared_edge_rtol_thres is not None and float(shared_edge_rtol_thres) >= 0:
            logging.info(
                f"Using shared_edge_rtol_thres={float(shared_edge_rtol_thres)} "
                f"for {self.__class__.__name__}"
            )
            return float(shared_edge_rtol_thres)

        if not enable_edge_rtol_thres or edge_rtol_thres is None:
            return 0.0
        return float(edge_rtol_thres)

    def __len__(self):
        return self.mixture_weight * self.samples_per_weight_unit

    def __getitem__(self, idx_N):
        """
        Get an item from the dataset.

        Args:
            idx_N: Tuple containing (seq_index, img_per_seq, aspect_ratio)

        Returns:
            Dataset item as returned by get_data()
        """
        seq_index, img_per_seq, aspect_ratio = idx_N

        # One unreadable sequence must not kill the run. Retry with a *different*
        # sequence: re-reading a permanently broken one never succeeds.
        #
        # - catch Exception, not OSError: read_npy_wrapper returns None on failure, so a
        #   bad pose/intrinsics file surfaces as RuntimeError further down
        # - img_per_seq and aspect_ratio are pinned per batch by DynamicBatchSampler and
        #   must never be re-rolled here, or default_collate will fail on mismatched shapes
        n_seq = getattr(self, "sequence_list_len", 0)
        max_attempts = 8 if n_seq > 1 else 1
        for attempt in range(max_attempts):
            try:
                return self.get_data(
                    seq_index=seq_index,
                    img_per_seq=img_per_seq,
                    aspect_ratio=aspect_ratio,
                )
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                logging.warning(
                    "%s: sample failed (attempt %d/%d, seq_index=%s): %r; resampling",
                    self.__class__.__name__,
                    attempt + 1,
                    max_attempts,
                    seq_index,
                    e,
                )
                time.sleep(random.uniform(0, min(2**attempt, 30)))
                seq_index = random.randrange(n_seq)

    def get_data(self, seq_index, img_per_seq, aspect_ratio):
        """
        Abstract method to retrieve data for a given sequence.

        Args:
            seq_index: Index of the sequence.
            img_per_seq: Number of frames to sample.
            aspect_ratio: Target aspect ratio.

        Returns:
            Dataset-specific data

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError(
            "This is an abstract method and should be implemented in the subclass, i.e., each dataset should implement its own get_data method."
        )

    def get_target_shape(self, aspect_ratio):
        """
        Calculate the target shape from the requested aspect ratio.

        By default this preserves the historical behavior of keeping the total
        token count approximately constant. When ``fixed_base_side`` is enabled,
        one image dimension is fixed to ``img_size`` and the other is derived
        from ``aspect_ratio``, then rounded to the nearest patch multiple.

        Args:
            aspect_ratio: Target aspect ratio (height / width or width / height)
            if_swap_height_width (bool): whether to swap height and width patches

        Returns:
            numpy.ndarray: Target image shape [height, width]
        """
        if aspect_ratio <= 0:
            raise ValueError(f"aspect_ratio must be positive, got {aspect_ratio}")

        if self.fixed_base_side:
            if aspect_ratio <= 1.0:
                height = self.img_size
                width = self._round_to_patch_multiple(self.img_size / aspect_ratio)
            else:
                width = self.img_size
                height = self._round_to_patch_multiple(self.img_size * aspect_ratio)
            return np.array([height, width], dtype=np.int32)

        token_number = self.token_number

        w_patches = np.sqrt(token_number / aspect_ratio)
        h_patches = token_number / w_patches
        w_patches = int(np.round(w_patches))
        h_patches = int(np.round(h_patches))

        # Calculate the final image shape in pixels.
        image_shape = np.array(
            [h_patches * self.patch_size, w_patches * self.patch_size], dtype=np.int32
        )
        return image_shape

    def _round_to_patch_multiple(self, value):
        """Round a side length to the nearest positive multiple of patch_size."""
        rounded = int(np.round(float(value) / self.patch_size)) * self.patch_size
        return max(self.patch_size, rounded)

    def _filter_depth_edges(self, depth_map, seq_name=None, idx=None):
        """Optionally zero out depth discontinuities detected from the GT depth map."""
        if depth_map is None or self.edge_rtol_thres <= 0:
            return depth_map

        valid_mask = np.isfinite(depth_map) & (depth_map > 0)
        if not np.any(valid_mask):
            return depth_map

        edge_mask = (
            depth_edge_opencv(
                depth_map,
                rtol=self.edge_rtol_thres,
                kernel_size=3,
                mask=valid_mask,
            )
            & valid_mask
        )
        if not np.any(edge_mask):
            return depth_map

        depth_map = np.array(depth_map, copy=True)
        depth_map[edge_mask] = 0.0

        if getattr(self, "verbose", False):
            logging.info(
                f"Masked {int(edge_mask.sum())} depth edge pixels for "
                f"{seq_name}/{idx} with edge_rtol_thres={self.edge_rtol_thres}"
            )
        return depth_map

    def sample_preprocess_params(self):
        """Sample the random preprocess decisions for a single image."""
        params = {
            "random_h_scale": 1.0,
            "random_w_scale": 1.0,
            "rot_aug_angle": None,
            "apply_rescale": self.rescale,
            "skip_extreme_rescale_gate": False,
            "rescale_safe_bound_ratio": 0.0,
        }

        if self.training and self.aug_scale:
            random_h_scale, random_w_scale = np.random.uniform(
                self.aug_scale[0], self.aug_scale[1], 2
            )
            if np.random.rand() < 0.25:
                random_w_scale = random_h_scale
            params["random_h_scale"] = min(float(random_h_scale), 1.0)
            params["random_w_scale"] = min(float(random_w_scale), 1.0)

        if (
            self.random_rotation_ratio > 0
            and random.random() < self.random_rotation_ratio
        ):
            params["rot_aug_angle"] = random.choice([90, 180, 270])

        if self.rescale and self.rescale_prob > 0:
            params["apply_rescale"] = random.random() < self.rescale_prob

        if params["apply_rescale"] and self.skip_extreme_rescale:
            params["skip_extreme_rescale_gate"] = random.random() > 0.3

        if params["apply_rescale"] and self.rescale_aug:
            params["rescale_safe_bound_ratio"] = float(np.random.triangular(0, 0, 0.3))

        return params

    @staticmethod
    def _validate_preprocess_params(preprocess_params):
        """Ensure externally provided preprocess params are complete."""
        required_keys = {
            "random_h_scale",
            "random_w_scale",
            "rot_aug_angle",
            "apply_rescale",
            "skip_extreme_rescale_gate",
            "rescale_safe_bound_ratio",
        }
        missing_keys = sorted(required_keys.difference(preprocess_params))
        if missing_keys:
            raise KeyError(
                "preprocess_params must be None or a complete dict. "
                f"Missing keys: {missing_keys}"
            )

    def process_one_image(
        self,
        image,
        depth_map,
        extri_opencv,
        intri_opencv,
        target_image_shape,
        filepath=None,
        safe_bound=4,
        preprocess_params=None,
        seq_name=None,
        frame_idx=None,
    ):
        """
        Process a single image and its associated data.

        This method handles image transformations, depth processing, and coordinate conversions.

        Args:
            image (numpy.ndarray): Input image array
            depth_map (numpy.ndarray): Depth map array
            extri_opencv (numpy.ndarray): Extrinsic camera matrix (OpenCV convention)
            intri_opencv (numpy.ndarray): Intrinsic camera matrix (OpenCV convention)
            target_image_shape (numpy.ndarray): Target image shape after processing
            filepath (str, optional): Optional file path for debugging. Defaults to None.
            safe_bound (int, optional): Safety margin for cropping operations. Defaults to 4.

        Returns:
            tuple: (
                image (numpy.ndarray): Processed image,
                depth_map (numpy.ndarray): Processed depth map,
                extri_opencv (numpy.ndarray): Updated extrinsic matrix,
                intri_opencv (numpy.ndarray): Updated intrinsic matrix,
                world_coords_points (numpy.ndarray): 3D points in world coordinates
            )
        """
        # Make copies to avoid in-place operations affecting original data
        image = np.copy(image)
        depth_map = np.copy(depth_map)
        extri_opencv = np.copy(extri_opencv)
        intri_opencv = np.copy(intri_opencv)
        original_size = np.array(image.shape[:2])

        if preprocess_params is None:
            preprocess_params = self.sample_preprocess_params()
        else:
            self._validate_preprocess_params(preprocess_params)

        # Apply random scale augmentation during training if enabled
        if self.training and self.aug_scale:
            random_h_scale = min(float(preprocess_params["random_h_scale"]), 1.0)
            random_w_scale = min(float(preprocess_params["random_w_scale"]), 1.0)
            aug_size = original_size * np.array([random_h_scale, random_w_scale])
            aug_size = aug_size.astype(np.int32)
        else:
            aug_size = original_size

        # Move principal point to the image center and crop if necessary
        image, depth_map, intri_opencv = crop_image_depth_and_intrinsic_by_pp(
            image,
            depth_map,
            intri_opencv,
            aug_size,
            filepath=filepath,
            aug_principal_point=self.aug_principal_point,
        )

        target_shape = target_image_shape

        # Apply random rotation augmentation
        rot_aug_angle = preprocess_params["rot_aug_angle"]
        if rot_aug_angle == 90 or rot_aug_angle == 270:
            target_shape = np.array([target_image_shape[1], target_image_shape[0]])

        # Resize images and update intrinsics
        # Determine whether to apply rescale for this sample
        apply_rescale = bool(preprocess_params["apply_rescale"])

        # Only allow no-rescale samples when the current crop can already cover the target.
        # Otherwise we would fall through to strict crop/pad and create large padded borders.
        if not apply_rescale:
            current_size = np.array(image.shape[:2])  # [H, W] after first crop
            if np.any(current_size < target_shape):
                apply_rescale = True

        # Optionally prevent any upsampling, regardless of skip_extreme_rescale
        if apply_rescale and self.avoid_upsample:
            current_size = np.array(image.shape[:2])  # [H, W] after first crop
            effective_target = target_shape + safe_bound
            scale_ratios = effective_target / current_size
            if np.max(scale_ratios) > 1.0:
                apply_rescale = False

        # Skip rescale if extreme scaling would be needed (to avoid depth map artifacts)
        # - Upsampling causes blocky artifacts with INTER_NEAREST interpolation
        # - Excessive downsampling (> max_downsample_ratio) causes loss of detail

        if (
            apply_rescale
            and self.skip_extreme_rescale
            and preprocess_params["skip_extreme_rescale_gate"]
        ):
            current_size = np.array(image.shape[:2])  # [H, W] after first crop

            # Use effective target size (including safe_bound) to match the actual logic in resize_image_depth_and_intrinsic
            # This prevents accidental upsampling when image is very close to target size
            effective_target = target_shape + safe_bound

            # Compute the scale ratio needed for each dimension
            # scale < 1.0 means downsampling, scale > 1.0 means upsampling
            scale_ratios = effective_target / current_size
            min_scale = np.min(scale_ratios)
            max_scale = np.max(scale_ratios)

            # Skip if any dimension needs upsampling (scale > 1.0)
            if max_scale > 1.0:
                apply_rescale = False
            # Skip if downsampling ratio exceeds threshold (e.g., scale < 1/1.5 ≈ 0.67)
            elif min_scale < (1.0 / self.max_downsample_ratio):
                apply_rescale = False

        if apply_rescale:
            image, depth_map, intri_opencv = resize_image_depth_and_intrinsic(
                image,
                depth_map,
                intri_opencv,
                target_shape,
                safe_bound=safe_bound,
                rescale_aug=self.rescale_aug,
                rescale_safe_bound_ratio=preprocess_params[
                    "rescale_safe_bound_ratio"
                ],
                shared_intrinsics_resize_scale=self.shared_intrinsics_resize_scale,
            )

        # Ensure final crop to target shape
        image, depth_map, intri_opencv = crop_image_depth_and_intrinsic_by_pp(
            image,
            depth_map,
            intri_opencv,
            target_shape,
            filepath=filepath,
            strict=True,
            aug_principal_point=self.aug_principal_point,
        )

        if rot_aug_angle is not None:
            if rot_aug_angle == 90:
                image, depth_map, extri_opencv, intri_opencv = rotate_90_degrees(
                    image,
                    depth_map,
                    extri_opencv,
                    intri_opencv,
                    clockwise=True,
                )
            elif rot_aug_angle == 180:
                # Rotate twice for 180 degrees
                for _ in range(2):
                    image, depth_map, extri_opencv, intri_opencv = (
                        rotate_90_degrees(
                            image,
                            depth_map,
                            extri_opencv,
                            intri_opencv,
                            clockwise=True,
                        )
                    )
            elif rot_aug_angle == 270:
                # 270 clockwise is the same as 90 counter-clockwise
                image, depth_map, extri_opencv, intri_opencv = rotate_90_degrees(
                    image,
                    depth_map,
                    extri_opencv,
                    intri_opencv,
                    clockwise=False,
                )

        # Filter GT depth edges on the final supervision grid after all spatial transforms.
        depth_map = self._filter_depth_edges(
            depth_map,
            seq_name=seq_name,
            idx=frame_idx,
        )

        # Convert depth to world coordinates.
        world_coords_points = depth_to_world_coords_points(
            depth_map, extri_opencv, intri_opencv
        )

        # Check if principal point is close to image center
        cx, cy = intri_opencv[0, 2], intri_opencv[1, 2]
        center_x, center_y = image.shape[1] / 2, image.shape[0] / 2
        pp_shift = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
        if pp_shift > 2:
            logging.warning(
                f"Principal point shift > 2 pixels ({pp_shift:.2f}): {filepath}"
            )

        return (
            image,
            depth_map,
            extri_opencv,
            intri_opencv,
            world_coords_points,
        )

    def get_nearby_ids_by_interval(
        self, ids, full_seq_num, interval=1, num_samples=None
    ):
        """
        Sample a set of IDs at fixed intervals centered on the anchor, adjusted for boundaries.

        Args:
            ids (list): Initial list of IDs. The first element is used as the anchor.
            full_seq_num (int): Total number of items in the full sequence.
            interval (int): Interval between sampled indices. Default is 1.
            num_samples (int, optional): Total number of IDs to sample including the anchor.
                                        If None, defaults to length of `ids`.

        Returns:
            numpy.ndarray: Array of sampled IDs, centered (as much as possible) on the anchor.

        Raises:
            ValueError: If no IDs are provided.
        """
        if len(ids) == 0:
            raise ValueError("No IDs provided.")

        if len(ids) * interval > full_seq_num:
            interval = full_seq_num // len(ids)

        if interval < 1:
            interval = 1

        if num_samples is None:
            num_samples = len(ids)

        anchor = ids[0]

        num_before = (num_samples - 1) // 2
        num_after = num_samples - 1 - num_before

        # Create the offsets.
        offsets = np.arange(-num_before, num_after + 1) * interval

        sampled = anchor + offsets

        # If out-of-bound, try to shift the window to stay in bounds
        min_id = sampled[0]
        max_id = sampled[-1]

        if min_id < 0:
            shift = -min_id
            sampled += shift
        elif max_id >= full_seq_num:
            shift = max_id - (full_seq_num - 1)
            sampled -= shift

        if random.random() < 0.1:
            # Only attempt to re-order if the anchor is still in the list.
            # If the shift logic pushed the anchor out, we skip this,
            # otherwise we would add it back and get (num_samples + 1) elements.
            if anchor in sampled:
                # [anchor, before, after]
                sampled = sampled[sampled != anchor]
                sampled = np.concatenate(([anchor], sampled))

        # Final clip to ensure all values are valid, in case the
        # shift was not enough or created a new out-of-bounds violation.
        sampled = np.clip(sampled, 0, full_seq_num - 1)

        return np.array(sampled, dtype=int)
