# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
import logging
import random
from pathlib import Path
import json

import cv2
import numpy as np
import glob


from data.dataset_util import (
    read_image_cv2,
    read_npy_wrapper,
    sort_ids,
    threshold_depth_map,
)
from data.base_dataset import BaseDataset
from data.datasets.unified_helper import (
    LazyIndexRanking,
    FRAME_COUNT_POWER_LAW_POLICY,
    FrameCountPowerLawConfig,
    REVISIT_POLICY_ALLOW,
    REVISIT_POLICY_AVOID_IF_POSSIBLE,
    resolve_frame_window,
    sample_frames_with_ranking,
    validate_dataset_frame_window_policy,
    validate_revisit_avoid_prob,
    validate_revisit_policy,
)


class UnifiedDataset(BaseDataset):
    """
    Unified dataset that loads data converted to the unified format.

    The unified format follows the structure defined in convert_eden.py:
    - Each scene directory contains: images/, depths/, image_names.json, cam_from_worlds.npy, intrinsics.npy, ranking.npy (optional)
    - image_names.json contains: list of image names, sorted
    - cam_from_worlds.npy contains: Nx3x4 array of extrinsics, OpenCV format, camera from world
    - intrinsics.npy contains: Nx3x3 array of intrinsics, OpenCV format, camera intrinsics
    - ranking.npy contains: NxN array of ranking (optional)
    - Images are named as XXXXX.ext (e.g., 00000.png, 00001.jpg, aabb.png, ccdd.png)
    - Depths are named as XXXXX.exr in EXR format
    """

    def __init__(
        self,
        common_conf,
        UNIFIED_DIR=None,
        mixture_weight=100,
        mixture_weight_scale=1.0,
        frame_window=8,
        min_interval=1,
        max_interval=-1,  # if max_interval, then take a probability to sample frames by interval, or else use ranking
        max_interval_ratio=0.025,
        sort_ratio=0.025,
        dataset_prefix="default",
        max_depth=-1,
        max_percentile=-1,
        min_percentile=-1,
        sequence_list_file=None,
        jump_prob=0.1,
        train_split_ratio=-1.0,
        sample_by_index=False,
        mask_path=None,
        per_frame_anno=False,
        image_names_filename="image_names.json",
        cam_from_worlds_filename="cam_from_worlds.npy",
        intrinsics_filename="intrinsics.npy",
        disable_depth_loss=False,
        min_valid_depth_ratio=0.01,
        edge_rtol_thres=0.0,
        valid_mask_erode_iters=-1,
        is_synthetic=False,
        force_crop=None,
        skip_first_ids=-1,
        frame_window_policy=None,
        revisit_policy=REVISIT_POLICY_ALLOW,
    ):
        super().__init__(common_conf=common_conf)

        self.min_valid_depth_ratio = min_valid_depth_ratio
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.verbose = getattr(common_conf, "verbose", False)
        self.enable_edge_rtol_thres = getattr(
            common_conf, "enable_edge_rtol_thres", False
        )
        self.synthetic_len_scale = getattr(common_conf, "synthetic_len_scale", 1)
        self.default_max_percentage_depth = getattr(
            common_conf, "default_max_percentage_depth", -1
        )

        self.frame_window = frame_window
        self.frame_window_policy = validate_dataset_frame_window_policy(
            frame_window_policy
        )
        self.frame_count_power_law_config = None
        if self.frame_window_policy == FRAME_COUNT_POWER_LAW_POLICY:
            self.frame_count_power_law_config = FrameCountPowerLawConfig.from_config(
                getattr(common_conf, "frame_count_power_law_defaults", None)
            )
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.max_interval_ratio = max_interval_ratio
        self.sort_ratio = sort_ratio
        self.jump_prob = jump_prob
        if self.min_interval < 1:
            raise ValueError(f"min_interval must be >= 1, got {self.min_interval}")
        if self.max_interval > 0 and self.min_interval > self.max_interval:
            raise ValueError(
                f"min_interval ({self.min_interval}) must be <= max_interval ({self.max_interval})"
            )
        self.UNIFIED_DIR = UNIFIED_DIR
        self.is_synthetic = is_synthetic
        if self.is_synthetic:
            mixture_weight = mixture_weight * self.synthetic_len_scale
        mixture_weight = int(mixture_weight * mixture_weight_scale)
        if mixture_weight < 1:
            mixture_weight = 1
        self.mixture_weight = mixture_weight
        self.name = "Unified" + dataset_prefix
        self.max_depth = max_depth
        if max_percentile < 0 and self.default_max_percentage_depth > 0:
            self.max_percentile = self.default_max_percentage_depth
        else:
            self.max_percentile = max_percentile
        self.min_percentile = min_percentile
        self.sequence_list_file = sequence_list_file
        self.train_split_ratio = train_split_ratio
        self.image_names_filename = image_names_filename
        self.cam_from_worlds_filename = cam_from_worlds_filename
        self.intrinsics_filename = intrinsics_filename
        self.disable_depth_loss = disable_depth_loss
        logging.info(
            f"{self.__class__.__name__} dataset_prefix={dataset_prefix} "
            f"disable_depth_loss={self.disable_depth_loss}"
        )
        self.edge_rtol_thres = self.resolve_edge_rtol_thres(
            edge_rtol_thres=edge_rtol_thres,
            enable_edge_rtol_thres=self.enable_edge_rtol_thres,
        )
        self.valid_mask_erode_iters = int(valid_mask_erode_iters)
        self.force_crop = self._parse_force_crop(force_crop)
        self.skip_first_ids = skip_first_ids
        if self.skip_first_ids > 0:
            assert sample_by_index, "skip_first_ids requires sample_by_index=True"

        self.per_frame_anno = per_frame_anno
        if self.per_frame_anno:
            if not sample_by_index:
                logging.warning(
                    "per_frame_anno=True requires sample_by_index=True. Overriding."
                )
                sample_by_index = True

        self.sample_by_index = sample_by_index
        self.revisit_policy = validate_revisit_policy(revisit_policy)
        self.revisit_avoid_prob = validate_revisit_avoid_prob(
            getattr(common_conf, "revisit_avoid_prob", 1.0)
        )
        if (
            self.revisit_policy == REVISIT_POLICY_AVOID_IF_POSSIBLE
            and self.revisit_avoid_prob < 1.0
        ):
            logging.info(
                f"Enabled stochastic revisit avoidance for {self.name}: "
                f"p={self.revisit_avoid_prob} per sampled sequence"
            )
        self.mask_path = mask_path
        if self.force_crop is not None and self.force_crop > 0:
            logging.info(
                f"Enabled force center crop for {self.name}: crop={self.force_crop}"
            )
        if self.edge_rtol_thres > 0:
            logging.info(
                f"Enabled depth edge filtering for {self.name}: "
                f"edge_rtol_thres={self.edge_rtol_thres}"
            )
        if self.valid_mask_erode_iters > 0:
            logging.info(
                f"Enabled valid depth mask erosion for {self.name}: "
                f"valid_mask_erode_iters={self.valid_mask_erode_iters}"
            )

        if not UNIFIED_DIR:
            raise ValueError(
                "UNIFIED_DIR must point at the root of a converted dataset"
            )

        logging.info(f"UNIFIED_DIR is {self.UNIFIED_DIR}")

        # Load sequence list
        if self.sequence_list_file:
            logging.info(f"Loading sequence list from {self.sequence_list_file}")
            with open(self.sequence_list_file, "r") as f:
                sequence_list = [line.strip() for line in f if line.strip()]
        else:
            # each directory in UNIFIED_DIR is a scene
            sequence_paths = glob.glob(osp.join(self.UNIFIED_DIR, "*"))
            sequence_paths = [p for p in sequence_paths if osp.isdir(p)]
            sequence_list = sorted(
                [osp.basename(seq.rstrip("/\\")) for seq in sequence_paths]
            )

        sequence_list = sorted(sequence_list)
        # Split dataset into training and testing sets
        if self.train_split_ratio > 0 and self.train_split_ratio < 1.0:
            logging.info(
                f"Splitting dataset {self.name} with train_split_ratio={self.train_split_ratio}, seed=42"
            )
            # Use a fixed seed for reproducibility
            rng = random.Random(42)
            rng.shuffle(sequence_list)

            split_index = int(len(sequence_list) * self.train_split_ratio)

            if self.training:
                sequence_list = sequence_list[:split_index]
            else:
                sequence_list = sequence_list[split_index:]

        self.sequence_list = tuple(sequence_list)
        self.sequence_list_len = len(self.sequence_list)

        if self.sequence_list_len == 0:
            raise ValueError(
                f"Dataset {self.name} is empty. UNIFIED_DIR: {self.UNIFIED_DIR}. Please check the path and permissions."
            )

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: {self.name} Real Data size: {self.sequence_list_len}")
        logging.info(f"{status}: {self.name} Data dataset length: {len(self)}")

    def resolve_frame_window(self, img_per_seq):
        return resolve_frame_window(
            self.frame_window,
            img_per_seq,
            frame_window_policy=self.frame_window_policy,
            low_n_config=self.frame_count_power_law_config,
        )

    def resolve_sample_revisit_policy(self):
        """Per-sequence coin for ``avoid_if_possible``.

        Keeps the configured policy with probability ``revisit_avoid_prob``,
        otherwise this one sampled sequence behaves as ``allow``. Both
        ``prob >= 1`` (the default) and ``prob <= 0`` draw no random number, so
        they reduce exactly to the two original policies.
        """
        if self.revisit_policy != REVISIT_POLICY_AVOID_IF_POSSIBLE:
            return self.revisit_policy
        if self.revisit_avoid_prob >= 1.0:
            return self.revisit_policy
        if self.revisit_avoid_prob <= 0.0 or random.random() >= self.revisit_avoid_prob:
            return REVISIT_POLICY_ALLOW
        return self.revisit_policy

    def get_data(
        self,
        seq_index=None,
        img_per_seq=None,
        seq_name=None,
        ids=None,
        aspect_ratio=1.0,
    ):
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        # Load scene metadata from separate files
        seq_path = osp.join(self.UNIFIED_DIR, seq_name)

        # Load image_names from JSON file
        image_names_path = osp.join(seq_path, self.image_names_filename)
        try:
            with open(image_names_path, "r") as f:
                image_names = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Error decoding JSON from {image_names_path}") from e

        num_imgs = len(image_names)

        if num_imgs < 3:
            logging.warning("Sequence %s has fewer than three images", seq_name)

        # Skip first frames if requested (only when ids are not externally provided)
        skip = 0
        if ids is None and self.skip_first_ids > 0:
            skip = min(self.skip_first_ids, max(0, num_imgs - 3))
            num_imgs = num_imgs - skip

        if ids is None:
            ids = np.random.choice(num_imgs, img_per_seq, replace=True)

        if img_per_seq > 1:
            effective_frame_window = self.resolve_frame_window(img_per_seq)

            if self.sample_by_index:
                ranking = LazyIndexRanking(num_imgs, effective_frame_window)
            else:
                # Load ranking from NPY file. If it fails, generate it.
                ranking_path = osp.join(seq_path, "ranking.npy")
                ranking = read_npy_wrapper(ranking_path, optional=True)

                if ranking is None:
                    logging.info(
                        f"Ranking not found at path: {ranking_path}. Generating on the fly."
                    )
                    ranking = LazyIndexRanking(num_imgs, effective_frame_window)

            ids = sample_frames_with_ranking(
                ids[0],
                ranking,
                img_per_seq,
                effective_frame_window,
                jump_prob=self.jump_prob,
                seq_name=seq_name,
                revisit_policy=self.resolve_sample_revisit_policy(),
            )

            if self.max_interval > 0 and random.random() < self.max_interval_ratio:
                interval = random.randint(self.min_interval, self.max_interval)
                ids = self.get_nearby_ids_by_interval(ids, num_imgs, interval=interval)

            # Sort IDs occasionally
            if random.random() < self.sort_ratio:
                ids = sort_ids(ids)

        # Restore original indices and num_imgs after skip
        if skip > 0:
            ids = ids + skip
            num_imgs = num_imgs + skip

        if not self.per_frame_anno:
            # Load cam_from_worlds from NPY file
            cam_from_worlds_path = osp.join(seq_path, self.cam_from_worlds_filename)
            cam_from_worlds = read_npy_wrapper(cam_from_worlds_path)  # Shape: (N, 3, 4)

            # Load intrinsics from NPY file
            intrinsics_path = osp.join(seq_path, self.intrinsics_filename)
            intrinsics = read_npy_wrapper(intrinsics_path)  # Shape: (N, 3, 3)

            if intrinsics is None:
                raise RuntimeError(
                    f"Failed to load intrinsics for sequence {seq_name} from {intrinsics_path}"
                )
            if cam_from_worlds is None:
                raise RuntimeError(
                    f"Failed to load cam_from_worlds for sequence {seq_name} from {cam_from_worlds_path}"
                )

            assert len(intrinsics) == num_imgs, (
                f"Sequence {seq_name} has mismatch between intrinsics ({len(intrinsics)}) and num_imgs ({num_imgs})"
            )
            assert len(cam_from_worlds) == num_imgs, (
                f"Sequence {seq_name} has mismatch between cam_from_worlds "
                f"({len(cam_from_worlds)}) and num_imgs ({num_imgs})"
            )
            self._validate_intrinsics_non_negative(intrinsics, seq_name=seq_name)
        else:
            cam_from_worlds = None
            intrinsics = None

        target_shape = self.get_target_shape(aspect_ratio)

        # containers
        images, depths = [], []
        world_points = []
        extrinsics, intrinsics_list = [], []
        sum_valid_depth_ratio = 0.0
        shared_preprocess_params = None
        if (
            self.same_preprocess_ratio >= 0
            and random.random() < self.same_preprocess_ratio
        ):
            shared_preprocess_params = self.sample_preprocess_params()

        for idx in ids:
            io_result = self._io_read(idx, seq_name, image_names)

            image, depth_map, img_path, depth_path, extri_frame, intri_frame = io_result

            image, depth_map = _validate_image_depth_pair(
                image, depth_map, img_path, depth_path, seq_name, idx
            )

            # Get camera parameters for this frame
            if self.per_frame_anno:
                intri = intri_frame.astype(np.float32)
                extri = extri_frame
            else:
                intri = intrinsics[idx].astype(np.float32)
                extri = cam_from_worlds[idx]  # Shape: (3, 4)

            if self.force_crop is not None:
                image, depth_map, intri = self._apply_force_center_crop(
                    image, depth_map, intri, seq_name=seq_name, idx=idx
                )

            if self.verbose:
                logging.info(
                    f"depth_map before thresholding {img_path} min {depth_map.min()} max {depth_map.max()}"
                )

            # Threshold depth map
            depth_map = threshold_depth_map(
                depth_map,
                max_percentile=self.max_percentile,
                min_percentile=self.min_percentile,
                max_depth=self.max_depth,
            )
            depth_map = self._erode_valid_depth_mask(
                depth_map, seq_name=seq_name, idx=idx
            )

            preprocess_params = (
                shared_preprocess_params
                if shared_preprocess_params is not None
                else self.sample_preprocess_params()
            )

            # Process the image through the base dataset pipeline
            (
                image,
                depth_map,
                extri,
                intri,
                world_pts,
            ) = self.process_one_image(
                image,
                depth_map,
                extri,
                intri,
                target_shape,
                filepath=img_path,
                preprocess_params=preprocess_params,
                seq_name=seq_name,
                frame_idx=idx,
            )

            if (image.shape[:2] != target_shape).any():
                # Skipping the frame would emit a sample with img_per_seq - 1
                # frames. Raise here so BaseDataset.__getitem__ resamples and the
                # error identifies the sequence that produced the wrong shape.
                raise ValueError(
                    f"Wrong shape for {seq_name} frame {idx}: "
                    f"expected {target_shape}, got {image.shape[:2]}"
                )

            if self.verbose:
                rot_det = np.linalg.det(extri[:3, :3])
                assert abs(rot_det - 1.0) < 1e-3, (
                    f"det(extri[:3,:3]) is {rot_det}, should be 1.0"
                )
                logging.info(f"det(extri[:3,:3]) {rot_det}")

            images.append(image)
            depths.append(depth_map)
            sum_valid_depth_ratio += (depth_map > 0).mean()
            extrinsics.append(extri)
            intrinsics_list.append(intri)
            world_points.append(world_pts)

        depth_train_mask = not self.disable_depth_loss
        if len(depths) > 0:
            avg_valid_ratio = sum_valid_depth_ratio / len(depths)
            if avg_valid_ratio < self.min_valid_depth_ratio:
                depth_train_mask = False

        batch = {
            "dataset": self.name,
            "seq_name": seq_name,
            "is_synthetic": self.is_synthetic,
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics_list,
            "world_points": world_points,
            "depth_train_mask": depth_train_mask,
        }

        return batch

    @staticmethod
    def _parse_force_crop(force_crop):
        """Parse and validate force crop setting as an integer border size."""
        if force_crop is None:
            return None

        if isinstance(force_crop, (int, np.integer)):
            if force_crop < 0:
                raise ValueError(f"force_crop must be non-negative, got {force_crop}")
            return int(force_crop)

        raise TypeError(f"force_crop must be None or int, got {type(force_crop)}")

    def _apply_force_center_crop(self, image, depth_map, intri, seq_name, idx):
        """
        Force crop a fixed border from all sides and update intrinsics accordingly.
        """
        crop = self.force_crop
        if crop == 0:
            return image, depth_map, intri

        h, w = image.shape[:2]
        max_crop_h = (h - 1) // 2
        max_crop_w = (w - 1) // 2
        if crop > max_crop_h or crop > max_crop_w:
            new_crop = min(crop, max_crop_h, max_crop_w)
            logging.warning(
                f"force_crop {self.force_crop} too large for frame {seq_name}/{idx} with shape {(h, w)}. "
                f"Clamping to {new_crop}."
            )
            crop = new_crop

        if crop == 0:
            return image, depth_map, intri

        top, bottom = crop, h - crop
        left, right = crop, w - crop

        image = image[top:bottom, left:right]
        if depth_map is not None:
            depth_map = depth_map[top:bottom, left:right]

        intri = np.copy(intri)
        intri[1, 2] = intri[1, 2] - top
        intri[0, 2] = intri[0, 2] - left
        return image, depth_map, intri

    def _erode_valid_depth_mask(self, depth_map, seq_name=None, idx=None):
        """Optionally erode the valid depth mask to invalidate boundary pixels."""
        if depth_map is None or self.valid_mask_erode_iters <= 0:
            return depth_map

        valid_mask = np.isfinite(depth_map) & (depth_map > 0)
        if not np.any(valid_mask):
            return depth_map

        eroded_valid_mask = cv2.erode(
            valid_mask.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=self.valid_mask_erode_iters,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        if np.array_equal(eroded_valid_mask, valid_mask):
            return depth_map

        depth_map = np.array(depth_map, copy=True)
        depth_map[~eroded_valid_mask] = 0.0

        if self.verbose:
            logging.info(
                f"Masked {int((valid_mask & ~eroded_valid_mask).sum())} boundary depth pixels "
                f"for {seq_name}/{idx} with valid_mask_erode_iters={self.valid_mask_erode_iters}"
            )
        return depth_map

    @staticmethod
    def _validate_intrinsics_non_negative(intrinsics, seq_name, idx=None):
        """Reject intrinsics containing negative values at load time."""
        if intrinsics is None:
            raise ValueError(f"Intrinsics are missing for sequence {seq_name}")

        if intrinsics.ndim == 2:
            min_value = float(np.min(intrinsics))
            if min_value < 0:
                frame_desc = f", frame {idx}" if idx is not None else ""
                raise ValueError(
                    f"Sequence {seq_name}{frame_desc} has negative intrinsic values "
                    f"(min={min_value})"
                )
            return

        if intrinsics.ndim == 3:
            min_values = np.min(intrinsics, axis=(1, 2))
            negative_indices = np.where(min_values < 0)[0]
            if negative_indices.size > 0:
                first_bad_idx = int(negative_indices[0])
                min_value = float(min_values[first_bad_idx])
                raise ValueError(
                    f"Sequence {seq_name} has negative intrinsic values in frame "
                    f"{first_bad_idx} (min={min_value})"
                )
            return

        raise ValueError(
            f"Unsupported intrinsics shape {intrinsics.shape} for sequence {seq_name}"
        )

    def _io_read(self, idx, seq_name, image_names):
        """
        Reads image and depth data for a single frame.
        """
        image_name = image_names[idx]
        img_path = osp.join(self.UNIFIED_DIR, seq_name, "images", image_name)

        # Construct depth path by replacing extension with .exr
        depth_name = Path(image_name).stem + ".exr"
        depth_path = osp.join(self.UNIFIED_DIR, seq_name, "depths", depth_name)

        # Read image
        image = read_image_cv2(img_path)

        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_map = self._apply_depth_zero_mask(
            depth_map=depth_map,
            mask_dir=self.mask_path,
            seq_name=seq_name,
            image_name=image_name,
        )

        extri, intri = None, None
        if self.per_frame_anno:
            frame_id = Path(image_name).stem
            extri_path = osp.join(
                self.UNIFIED_DIR, seq_name, "cam_from_worlds", f"{frame_id}.npy"
            )
            intri_path = osp.join(
                self.UNIFIED_DIR, seq_name, "intrinsics", f"{frame_id}.npy"
            )
            extri = read_npy_wrapper(extri_path)
            intri = read_npy_wrapper(intri_path)
            self._validate_intrinsics_non_negative(
                intri, seq_name=seq_name, idx=frame_id
            )

        return image, depth_map, img_path, depth_path, extri, intri

    def _apply_depth_zero_mask(self, depth_map, mask_dir, seq_name, image_name):
        """Zero depth where the binary mask marks pixels as invalid."""
        if depth_map is None or not mask_dir:
            return depth_map

        mask_name = Path(image_name).stem + ".png"
        mask_path = osp.join(self.UNIFIED_DIR, seq_name, mask_dir, mask_name)
        try:
            mask = read_image_cv2(mask_path, rgb=False, optional=True)
        except (FileNotFoundError, OSError) as e:
            logging.warning(
                f"Configured mask not found or unreadable: {mask_path} ({e})"
            )
            return depth_map
        if mask is None:
            logging.warning(f"Configured mask not found or unreadable: {mask_path}")
            return depth_map

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        if mask.shape != depth_map.shape:
            mask = cv2.resize(
                mask,
                (depth_map.shape[1], depth_map.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        invalid_mask = mask == 0
        if not np.any(invalid_mask):
            return depth_map

        if np.all(invalid_mask):
            return np.zeros_like(depth_map)

        return np.where(invalid_mask, 0.0, depth_map)


def _validate_image_depth_pair(image, depth_map, img_path, depth_path, seq_name, idx):
    """Handle missing image/depth and ensure they have matching shapes."""
    if image is None and depth_map is None:
        logging.error(
            f"Image and depth are both None. Image path: {img_path}, Depth path: {depth_path}"
        )
        raise ValueError(
            f"Image and depth are both None for sequence {seq_name}, index {idx}"
        )

    if image is None:
        logging.warning(f"Image is None at path: {img_path}. Creating a black image.")
        image = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)

    if depth_map is None:
        logging.warning(
            f"Depth map is None at path: {depth_path}. Creating a zero depth map."
        )
        depth_map = np.zeros(image.shape[:2], dtype=np.float32)

    if image.shape[:2] != depth_map.shape:
        logging.warning(
            f"Image shape {image.shape[:2]} and depth shape {depth_map.shape} mismatch. "
            f"Resizing image for {img_path} to match depth map from {depth_path}"
        )
        image = cv2.resize(
            image,
            (depth_map.shape[1], depth_map.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

    return image, depth_map
