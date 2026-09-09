# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

import torch

from vggt_omega.utils.geometry import closed_form_inverse_se3


def _format_dataset_seq_meta(dataset_name=None, seq_name=None) -> str:
    if dataset_name is None and seq_name is None:
        return ""
    return f" dataset={dataset_name}, seq_name={seq_name},"


def check_valid_tensor(
    input_tensor: Optional[torch.Tensor],
    name: str = "tensor",
    context: str = "",
    dataset_name=None,
    seq_name=None,
) -> None:
    """
    Check if a tensor contains NaN/Inf/large values and print a warning if found.

    Args:
        input_tensor: The tensor to check
        name: Name of the tensor for logging purposes
        context: Context string indicating the calling function (e.g., "_normalize_scene_differentiable")
    """
    if input_tensor is not None:
        if input_tensor.dtype == torch.bool:
            return

        has_nan = torch.isnan(input_tensor).any()
        has_inf = torch.isinf(input_tensor).any()
        has_large = (input_tensor.abs() > 1e6).any()

        if has_nan or has_inf or has_large:
            ctx_str = f"[{context}] " if context else ""
            issues = []
            if has_nan:
                nan_count = torch.isnan(input_tensor).sum().item()
                issues.append(f"NaN count: {nan_count}")
            if has_inf:
                inf_count = torch.isinf(input_tensor).sum().item()
                issues.append(f"Inf count: {inf_count}")
            if has_large:
                large_count = (input_tensor.abs() > 1e6).sum().item()
                max_val = input_tensor.abs().max().item()
                issues.append(f"Large values (>1e6) count: {large_count}, max abs: {max_val:.2e}")

            # Get valid statistics (excluding NaN/Inf)
            valid_mask = ~(torch.isnan(input_tensor) | torch.isinf(input_tensor))
            if valid_mask.any():
                valid_vals = input_tensor[valid_mask]
                stats_str = f"valid range: [{valid_vals.min().item():.4f}, {valid_vals.max().item():.4f}], mean: {valid_vals.mean().item():.4f}"
            else:
                stats_str = "no valid values"

            meta_str = _format_dataset_seq_meta(dataset_name=dataset_name, seq_name=seq_name)
            logging.warning(
                f"{ctx_str}Invalid tensor '{name}':{meta_str} shape={list(input_tensor.shape)}, "
                f"dtype={input_tensor.dtype}, {', '.join(issues)}, {stats_str}"
            )


def _normalize_scene_differentiable(
    extrinsics: torch.Tensor,
    world_points: torch.Tensor,
    depths: torch.Tensor,
    point_masks: torch.Tensor,
    scale_only: bool = False,
    rel_to_first_cam: bool = True,
    context: str = "",
    dataset_name=None,
    seq_name=None,
):
    """Compute scene scale and optionally return normalized tensors."""
    ctx = f"_normalize_scene_differentiable ({context})" if context else "_normalize_scene_differentiable"
    check_valid_tensor(extrinsics, "extrinsics (input)", ctx, dataset_name=dataset_name, seq_name=seq_name)
    check_valid_tensor(world_points, "world_points (input)", ctx, dataset_name=dataset_name, seq_name=seq_name)
    check_valid_tensor(depths, "depths (input)", ctx, dataset_name=dataset_name, seq_name=seq_name)
    check_valid_tensor(point_masks, "point_masks (input)", ctx, dataset_name=dataset_name, seq_name=seq_name)

    if rel_to_first_cam:
        R = extrinsics[:, 0, :3, :3].float()
        t = extrinsics[:, 0, :3, 3].float()
        new_world_points = (
            world_points @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)
        ) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    else:
        new_world_points = world_points

    dist = new_world_points.norm(dim=-1)
    valid_count = point_masks.sum(dim=(1, 2, 3))

    # Avoid invalid values on the branch that torch.where later masks out.
    valid_count_safe = torch.clamp(valid_count, min=1.0)

    dist_sum = (dist * point_masks).sum(dim=(1, 2, 3))
    avg_scale = dist_sum / valid_count_safe

    # Guard against nan, inf, and tiny scales
    is_invalid_scale = (avg_scale < 1e-6) | torch.isnan(avg_scale) | torch.isinf(avg_scale)
    avg_scale = torch.where(is_invalid_scale, torch.ones_like(avg_scale), avg_scale)

    # Use torch.where instead of index assignment for differentiability
    # This also handles the case where valid_count <= 100 (not enough valid points)
    valid_seq_mask = (valid_count > 100) & (~is_invalid_scale)
    avg_scale = torch.where(valid_seq_mask, avg_scale, torch.ones_like(avg_scale))

    if scale_only:
        return avg_scale, valid_seq_mask

    batch_size, num_frames = extrinsics.shape[:2]
    last_row = torch.zeros(
        (batch_size, num_frames, 1, 4),
        device=extrinsics.device,
        dtype=extrinsics.dtype,
    )
    last_row[..., 3] = 1.0
    extrinsics_homog = torch.cat([extrinsics, last_row], dim=-2)
    if rel_to_first_cam:
        first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
        new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))
    else:
        new_extrinsics = extrinsics_homog

    new_depths = depths

    # Scale things
    # avg_scale is (B,)
    scale_view_5 = avg_scale.view(-1, 1, 1, 1, 1)
    scale_view_3 = avg_scale.view(-1, 1, 1)
    scale_view_4 = avg_scale.view(-1, 1, 1, 1)

    new_world_points = new_world_points / scale_view_5

    # new_extrinsics translation part
    # Construct new_extrinsics without in-place modification
    T_scaled = new_extrinsics[..., :3, 3] / scale_view_3
    new_extrinsics_R = new_extrinsics[..., :3, :3]
    new_extrinsics_bottom = new_extrinsics[..., 3:, :]  # 0 0 0 1
    new_extrinsics = torch.cat(
        [
            torch.cat([new_extrinsics_R, T_scaled.unsqueeze(-1)], dim=-1),
            new_extrinsics_bottom,
        ],
        dim=-2,
    )

    new_depths = new_depths / scale_view_4  # B, S, H, W
    new_extrinsics = new_extrinsics[:, :, :3]  # 4x4 -> 3x4

    check_valid_tensor(new_extrinsics, "extrinsics (output)", ctx, dataset_name=dataset_name, seq_name=seq_name)
    check_valid_tensor(new_world_points, "world_points (output)", ctx, dataset_name=dataset_name, seq_name=seq_name)
    check_valid_tensor(new_depths, "depths (output)", ctx, dataset_name=dataset_name, seq_name=seq_name)
    check_valid_tensor(avg_scale, "avg_scale", ctx, dataset_name=dataset_name, seq_name=seq_name)

    return new_extrinsics, new_world_points, new_depths, valid_seq_mask


def normalize_camera_extrinsics_and_points_batch(batch: dict) -> dict:
    """Normalize camera extrinsics and corresponding 3D points.

    This function transforms the coordinate system to be centered at the first camera
    and optionally scales the scene to have unit average distance.

    Supports both batched and non-batched inputs:
    - Batched: extrinsics (B, S, 3, 4), world_points (B, S, H, W, 3), etc.
    - Non-batched: extrinsics (S, 3, 4), world_points (S, H, W, 3), etc.

    Args:
        batch: Dictionary containing:
            - extrinsics: Camera extrinsic matrices of shape (B, S, 3, 4) or (S, 3, 4)
            - world_points: 3D points in world coordinates of shape (B, S, H, W, 3) or (S, H, W, 3)
            - depths: Depth maps of shape (B, S, H, W) or (S, H, W)
            - point_masks: Boolean masks for valid points of shape (B, S, H, W) or (S, H, W)

    Returns:
        Updated batch dictionary with normalized values (same shape as inputs)
    """
    extrinsics = batch["extrinsics"]
    world_points = batch["world_points"]
    depths = batch["depths"]
    point_masks = batch["point_masks"]
    dataset_name = batch.get("dataset")
    seq_name = batch.get("seq_name")

    # Check if inputs are batched or not
    # Batched: extrinsics is (B, S, 3, 4)
    # Non-batched: extrinsics is (S, 3, 4)
    is_batched = extrinsics.ndim == 4

    if not is_batched:
        # Add batch dimension
        extrinsics = extrinsics.unsqueeze(0)
        world_points = world_points.unsqueeze(0)
        depths = depths.unsqueeze(0)
        point_masks = point_masks.unsqueeze(0)

    with torch.no_grad():
        new_extrinsics, new_world_points, new_depths, valid_seq_mask = (
            _normalize_scene_differentiable(
                extrinsics,
                world_points,
                depths,
                point_masks,
                rel_to_first_cam=True,
                context="gt processing",
                dataset_name=dataset_name,
                seq_name=seq_name,
            )
        )

    if not is_batched:
        # Remove batch dimension
        new_extrinsics = new_extrinsics[0]
        new_world_points = new_world_points[0]
        new_depths = new_depths[0]
        valid_seq_mask = valid_seq_mask[0]
    if "valid_seq_mask" in batch:
        valid_seq_mask = torch.logical_and(valid_seq_mask, batch["valid_seq_mask"])
    batch["valid_seq_mask"] = valid_seq_mask
    batch["extrinsics"] = new_extrinsics
    batch["world_points"] = new_world_points
    batch["depths"] = new_depths

    return batch
