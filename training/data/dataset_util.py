# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import random

import cv2
import numpy as np
import torch

from vggt_omega.utils.geometry import closed_form_inverse_se3


# Geometric functions: cropping, resizing, coordinate transforms with camera intrinsics/extrinsics.
def crop_image_depth_and_intrinsic_by_pp(
    image,
    depth_map,
    intrinsic,
    target_shape,
    filepath=None,
    strict=False,
    aug_principal_point=False,
):
    """
    Crops the given image and depth map around the camera's principal point, as defined by `intrinsic`.
    Specifically:
      - Ensures that the crop is centered on (cx, cy).
      - Optionally pads the image (and depth map) if `strict=True` and the result is smaller than `target_shape`.
      - Shifts the camera intrinsic matrix accordingly.

    Args:
        image (np.ndarray):
            Input image array of shape (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map array of shape (H, W), or None if not available.
        intrinsic (np.ndarray):
            Camera intrinsic matrix (3x3). The principal point is assumed to be at (intrinsic[1,2], intrinsic[0,2]).
        target_shape (tuple[int, int]):
            Desired output shape.
        filepath (str or None):
            An optional file path for debug logging (only used if strict mode triggers warnings).
        strict (bool):
            If True, will zero-pad to ensure the exact target_shape even if the cropped region is smaller.

    Raises:
        AssertionError:
            If the input image is smaller than `target_shape`.
        ValueError:
            If the cropped image is larger than `target_shape` (in strict mode), which should not normally happen.

    Returns:
        tuple:
            (cropped_image, cropped_depth_map, updated_intrinsic)

            - cropped_image (np.ndarray): Cropped (and optionally padded) image.
            - cropped_depth_map (np.ndarray or None): Cropped (and optionally padded) depth map.
            - updated_intrinsic (np.ndarray): Intrinsic matrix adjusted for the crop.
    """
    original_size = np.array(image.shape)
    intrinsic = np.copy(intrinsic)

    if original_size[0] < 0.5 * target_shape[0]:
        error_message = (
            f"Height check failed: original height {original_size[0]} "
            f"is less than target height {target_shape[0]}. "
            f"File: {filepath}"
        )
        logging.warning(error_message)

    if original_size[1] < 0.5 * target_shape[1]:
        error_message = (
            f"Width check failed: original width {original_size[1]} "
            f"is less than target width {target_shape[1]}. "
            f"File: {filepath}"
        )
        logging.warning(error_message)

    # Identify principal point (center_h, center_w) from intrinsic
    center_h = intrinsic[1, 2]
    center_w = intrinsic[0, 2]

    # Compute how far we can crop in each direction
    if strict:
        half_h = min((target_shape[0] / 2), center_h)
        half_w = min((target_shape[1] / 2), center_w)
    else:
        half_h = min((target_shape[0] / 2), center_h, original_size[0] - center_h)
        half_w = min((target_shape[1] / 2), center_w, original_size[1] - center_w)

    if aug_principal_point:
        # NOTE: do we want to further aug this?
        start_h = int(round(center_h - half_h))
        start_w = int(round(center_w - half_w))
    else:
        # Compute starting indices
        start_h = math.floor(center_h) - math.floor(half_h)
        start_w = math.floor(center_w) - math.floor(half_w)

    # Clamp start indices to valid range (instead of assert)
    start_h = max(0, start_h)
    start_w = max(0, start_w)

    # Compute ending indices
    if strict:
        end_h = start_h + target_shape[0]
        end_w = start_w + target_shape[1]
    else:
        end_h = start_h + 2 * math.floor(half_h)
        end_w = start_w + 2 * math.floor(half_w)

    # Clamp end indices to image boundaries
    end_h = min(end_h, original_size[0])
    end_w = min(end_w, original_size[1])

    # Perform the crop
    image = image[start_h:end_h, start_w:end_w, :]
    if depth_map is not None:
        depth_map = depth_map[start_h:end_h, start_w:end_w]

    # Shift the principal point in the intrinsic
    intrinsic[1, 2] = intrinsic[1, 2] - start_h
    intrinsic[0, 2] = intrinsic[0, 2] - start_w

    # If strict, zero-pad if the new shape is smaller than target_shape
    if strict:
        if (image.shape[:2] != target_shape).any():
            current_h, current_w = image.shape[:2]
            target_h, target_w = target_shape[0], target_shape[1]
            pad_h = target_h - current_h
            pad_w = target_w - current_w
            if pad_h < 0 or pad_w < 0:
                raise ValueError(
                    f"The cropped image is bigger than the target shape: "
                    f"cropped=({current_h},{current_w}), "
                    f"target=({target_h},{target_w})."
                )

            # Calculate padding to place PP exactly at the center of the new image
            # Target PP position (center of new image)
            target_cy = target_h / 2.0
            target_cx = target_w / 2.0

            # Current PP position
            current_cy = intrinsic[1, 2]
            current_cx = intrinsic[0, 2]

            # Calculate ideal padding to center PP
            ideal_pad_top = target_cy - current_cy
            ideal_pad_left = target_cx - current_cx

            # Round to integers
            pad_top = int(round(ideal_pad_top))
            pad_left = int(round(ideal_pad_left))

            # Clamp to valid range [0, pad_h] and [0, pad_w]
            pad_top = max(0, min(pad_top, pad_h))
            pad_left = max(0, min(pad_left, pad_w))

            pad_bottom = pad_h - pad_top
            pad_right = pad_w - pad_left

            image = np.pad(
                image,
                pad_width=((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            if depth_map is not None:
                depth_map = np.pad(
                    depth_map,
                    pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode="constant",
                    constant_values=0,
                )
            # Update the intrinsic using the actual padding amounts.
            # Note: PP will be at most 0.5 pixels off from exact center due to integer padding
            intrinsic[1, 2] = current_cy + pad_top
            intrinsic[0, 2] = current_cx + pad_left

    return image, depth_map, intrinsic


def resize_image_depth_and_intrinsic(
    image,
    depth_map,
    intrinsic,
    target_shape,
    safe_bound=4,
    rescale_aug=True,
    rescale_safe_bound_ratio=None,
    shared_intrinsics_resize_scale=False,
):
    """
    Resizes the given image and depth map (if provided) to slightly larger than `target_shape`,
    updating the intrinsic matrix. Optionally uses random rescaling
    to create some additional margin (based on `rescale_aug`).

    Steps:
      1. Compute a scaling factor so that the resized result is at least `target_shape + safe_bound`.
      2. Apply an optional triangular random factor if `rescale_aug=True`.
      3. Resize the image with OpenCV Lanczos if downscaling, cubic if upscaling.
      4. Resize the depth map with nearest-neighbor.
      5. Update the camera intrinsic.

    Args:
        image (np.ndarray):
            Input image array (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map array (H, W), or None if unavailable.
        intrinsic (np.ndarray):
            Camera intrinsic matrix (3x3).
        target_shape (np.ndarray or tuple[int, int]):
            Desired final shape (height, width).
        safe_bound (int or float):
            Additional margin (in pixels) to add to target_shape before resizing.
        rescale_aug (bool):
            If True, randomly increase the `safe_bound` within a certain range to simulate augmentation.
        shared_intrinsics_resize_scale (bool):
            If True, update intrinsics with a single shared
            scale derived from the resized output. If False, use separate
            scale_x/scale_y updates.

    Returns:
        tuple:
            (resized_image, resized_depth_map, updated_intrinsic)

            - resized_image (np.ndarray): The resized image.
            - resized_depth_map (np.ndarray or None): The resized depth map.
            - updated_intrinsic (np.ndarray): Camera intrinsic updated for new resolution.

    Raises:
        AssertionError:
            If the shapes of the resized image and depth map do not match.
    """
    if rescale_aug:
        if rescale_safe_bound_ratio is None:
            random_boundary = np.random.triangular(0, 0, 0.3)
        else:
            random_boundary = float(rescale_safe_bound_ratio)
        safe_bound = safe_bound + random_boundary * target_shape.max()

    original_size = np.array(image.shape[:2])
    resize_scales = (target_shape + safe_bound) / original_size
    max_resize_scale = np.max(resize_scales)
    intrinsic = np.copy(intrinsic)

    # Compute output resolution
    input_resolution = np.array([image.shape[1], image.shape[0]])  # (width, height)
    output_resolution = np.floor(input_resolution * max_resize_scale).astype(int)

    interpolation = cv2.INTER_LANCZOS4 if max_resize_scale < 1 else cv2.INTER_CUBIC
    image = cv2.resize(image, tuple(output_resolution), interpolation=interpolation)

    if depth_map is not None:
        depth_map = cv2.resize(
            depth_map,
            output_resolution,
            fx=max_resize_scale,
            fy=max_resize_scale,
            interpolation=cv2.INTER_NEAREST,
        )

    actual_h, actual_w = image.shape[:2]
    original_h, original_w = original_size

    scale_y = actual_h / original_h
    scale_x = actual_w / original_w
    if shared_intrinsics_resize_scale:
        shared_scale = max(scale_y, scale_x)
        scale_y = shared_scale
        scale_x = shared_scale

    intrinsic[0, :] *= scale_x
    intrinsic[1, :] *= scale_y

    assert image.shape[:2] == depth_map.shape[:2]
    return image, depth_map, intrinsic


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    """
    Converts a depth map to world coordinates (HxWx3) given the camera extrinsic and intrinsic.

    Args:
        depth_map (np.ndarray):
            Depth map of shape (H, W).
        extrinsic (np.ndarray):
            Extrinsic matrix of shape (3, 4), representing the camera pose in OpenCV convention (camera-from-world).
        intrinsic (np.ndarray):
            Intrinsic matrix of shape (3, 3).

    Returns:
        np.ndarray: (H, W, 3) array of 3D points in world frame.
    """
    if depth_map is None:
        return None

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # The extrinsic is camera-from-world, so invert it to transform camera->world
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]
    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = (
        np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world
    )  # HxWx3, 3x3 -> HxWx3

    return world_coords_points


def depth_to_cam_coords_points(
    depth_map: np.ndarray, intrinsic: np.ndarray
) -> np.ndarray:
    """
    Unprojects a depth map into camera coordinates, returning (H, W, 3).

    Args:
        depth_map (np.ndarray):
            Depth map of shape (H, W).
        intrinsic (np.ndarray):
            3x3 camera intrinsic matrix.
            Assumes zero skew and standard OpenCV layout:
            [ fx   0   cx ]  # W
            [  0  fy   cy ]  # H
            [  0   0    1 ]

    Returns:
        np.ndarray:
            An (H, W, 3) array, where each pixel is mapped to (x, y, z) in the camera frame.
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, (
        "Intrinsic matrix must have zero skew"
    )

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    u = u + 0.5
    v = v + 0.5

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    return np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)


def depth_edge_opencv(
    depth: np.ndarray,
    atol: float = None,
    rtol: float = None,
    kernel_size: int = 3,
    mask: np.ndarray = None,
) -> np.ndarray:
    """
    OpenCV-backed depth edge detector used before the gradient loss.

    Args:
        depth: Array of shape (..., H, W) containing linear depth.
        atol: Absolute tolerance for edge detection.
        rtol: Relative tolerance for edge detection.
        kernel_size: Neighborhood size used to compute the local depth range.
        mask: Optional validity mask with the same shape as `depth`. If None,
            defaults to `depth > 0`.

    Returns:
        Boolean edge mask with the same shape as `depth`.
    """
    depth = np.asarray(depth)
    if depth.ndim < 2:
        raise ValueError(f"depth must have shape (..., H, W), got {depth.shape}")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd integer, got {kernel_size}"
        )

    if not np.issubdtype(depth.dtype, np.floating):
        depth = depth.astype(np.float32, copy=False)

    shape = depth.shape
    if mask is None:
        mask = depth > 0
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError(f"mask shape {mask.shape} must match depth shape {shape}")

    depth = depth.reshape(-1, *shape[-2:])
    mask = mask.reshape(-1, *shape[-2:])

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    neg_fill = np.finfo(depth.dtype).min
    pos_fill = np.finfo(depth.dtype).max
    edge = np.zeros_like(depth, dtype=bool)
    for i in range(depth.shape[0]):
        depth_i = depth[i]
        mask_i = mask[i]

        valid_any = cv2.dilate(
            mask_i.astype(np.uint8),
            kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)

        masked_for_max = np.where(mask_i, depth_i, neg_fill)
        masked_for_min = np.where(mask_i, depth_i, pos_fill)

        local_max = cv2.dilate(
            masked_for_max,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=float(neg_fill),
        )
        local_min = cv2.erode(
            masked_for_min,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=float(pos_fill),
        )

        diff = np.zeros_like(depth_i)
        np.subtract(local_max, local_min, out=diff, where=valid_any)

        edge_i = np.zeros_like(depth_i, dtype=bool)
        if atol is not None:
            edge_i |= diff > atol
        if rtol is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                edge_i |= np.nan_to_num(diff / depth_i) > rtol
        edge[i] = edge_i

    return edge.reshape(shape)


# I/O and non-geometric functions.
_npy_error_count = 0


def _log_npy_error(path, exc):
    """Rate-limited: BaseDataset.__getitem__ retries a failed sample several times, per
    worker, per rank, so an uncapped log is louder than the failure it reports."""
    global _npy_error_count
    _npy_error_count += 1
    if _npy_error_count <= 20 or _npy_error_count % 1000 == 0:
        logging.error(
            "Error loading numpy array (failure #%d) from %s: %r",
            _npy_error_count,
            path,
            exc,
        )


def read_npy_wrapper(path, optional: bool = False, **kwargs):
    """Returns None instead of raising, so a missing file degrades rather than kills a run.

    Set optional=True for files that are allowed to be absent, such as ranking.npy: those
    return None without logging, since the caller has a documented fallback.
    """
    try:
        return np.load(path, **kwargs)
    except Exception as e:
        if not (optional and isinstance(e, FileNotFoundError)):
            _log_npy_error(path, e)
        return None


def read_image_cv2(path: str, rgb: bool = True, optional: bool = False) -> np.ndarray:
    """Read an image as RGB (H, W, 3), or None if it cannot be read.

    Set optional=True for files that are allowed to be absent, such as masks: those return
    None quietly. A missing required frame is retried once and reported.
    """
    img = cv2.imread(path)
    if img is None:
        if optional:
            return None
        logging.warning(f"Could not load image={path}. Retrying...")
        img = cv2.imread(path)
        if img is None:
            logging.warning(f"Retry failed for image={path}.")
            return None

    if rgb:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def sort_ids(ids):
    if random.random() < 0.5:
        # forward sort
        return np.sort(ids)
    else:
        # reverse sort
        return np.sort(ids)[::-1]


def apply_random_patch_mask(images, depth_maps, random_patch_mask_size):
    """
    Apply a random patch mask to a batch of images and their corresponding depth maps.

    Args:
        images (torch.Tensor): A batch of image tensors of shape (B, 3, H, W).
        depth_maps (torch.Tensor): A batch of depth map tensors of shape (B, H, W).
        random_patch_mask_size (int): The maximum size of the patch mask.

    Returns:
        tuple: A tuple containing the masked images and depth maps.
    """
    B, _, H, W = images.shape
    device = images.device

    # Generate random mask sizes
    mask_size_h = torch.randint(0, random_patch_mask_size, (B,), device=device)
    mask_size_w = torch.randint(0, random_patch_mask_size, (B,), device=device)

    # Generate random top-left corners for the masks
    max_top = H - mask_size_h
    max_left = W - mask_size_w
    top = (torch.rand(B, device=device) * max_top).to(torch.long)
    left = (torch.rand(B, device=device) * max_left).to(torch.long)

    # Create coordinate grids
    h_indices = torch.arange(H, device=device).view(1, H, 1).expand(B, -1, W)
    w_indices = torch.arange(W, device=device).view(1, 1, W).expand(B, H, -1)

    # Expand mask dimensions for broadcasting
    top_expanded = top.view(B, 1, 1)
    left_expanded = left.view(B, 1, 1)
    bottom_expanded = (top + mask_size_h).view(B, 1, 1)
    right_expanded = (left + mask_size_w).view(B, 1, 1)

    # Create the boolean mask
    mask = (
        (h_indices >= top_expanded)
        & (h_indices < bottom_expanded)
        & (w_indices >= left_expanded)
        & (w_indices < right_expanded)
    )

    # Apply mask to images with random fill values (0 or 1)
    fill_values = torch.randint(0, 2, (B, 1, 1, 1), device=device, dtype=images.dtype)
    image_mask = mask.unsqueeze(1).expand_as(images)
    images = torch.where(image_mask, fill_values, images)

    # Apply mask to depth maps with a fill value of 0
    depth_mask = mask
    depth_maps = depth_maps.masked_fill(depth_mask, 0)

    return images, depth_maps


def threshold_depth_map(
    depth_map: np.ndarray,
    max_percentile: float = 99,
    min_percentile: float = 1,
    max_depth: float = -1,
) -> np.ndarray:
    """
    Thresholds a depth map using percentile-based limits and optional maximum depth clamping.

    Steps:
      1. If `max_depth > 0`, clamp all values above `max_depth` to zero.
      2. Compute `max_percentile` and `min_percentile` thresholds using nanpercentile.
      3. Zero out values above/below these thresholds, if thresholds are > 0.

    Args:
        depth_map (np.ndarray):
            Input depth map (H, W).
        max_percentile (float):
            Upper percentile (0-100). Values above this will be set to zero.
        min_percentile (float):
            Lower percentile (0-100). Values below this will be set to zero.
        max_depth (float):
            Absolute maximum depth. If > 0, any depth above this is set to zero.
            If <= 0, no maximum-depth clamp is applied.

    Returns:
        np.ndarray:
            Depth map (H, W) after thresholding. Some or all values may be zero.
            Returns None if depth_map is None.
    """
    if depth_map is None:
        return None
    depth_map = depth_map.astype(float, copy=True)

    depth_map[(depth_map < 0.0) | np.isinf(depth_map) | np.isnan(depth_map)] = 0.0

    # Optional clamp by max_depth
    if max_depth > 0:
        depth_map[depth_map > max_depth] = 0.0

    # Percentile-based thresholds
    depth_max_thres = (
        np.nanpercentile(depth_map, max_percentile) if max_percentile > 0 else None
    )
    depth_min_thres = (
        np.nanpercentile(depth_map, min_percentile) if min_percentile > 0 else None
    )

    # Apply the thresholds if they are > 0
    if depth_max_thres is not None and depth_max_thres > 0:
        depth_map[depth_map >= depth_max_thres] = 0.0
    if depth_min_thres is not None and depth_min_thres > 0:
        depth_map[depth_map <= depth_min_thres] = 0.0

    return depth_map
