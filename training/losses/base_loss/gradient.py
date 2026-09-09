# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from train_utils.general import check_and_fix_inf_nan


def gradient_loss_config_wrapper(
    prediction,
    target,
    mask,
    gt_depth,
    config=None,
):
    """Compute the weighted auxiliary losses declared in the loss config."""
    if config is None:
        return 0
    if gt_depth.ndim == 5:
        gt_depth = gt_depth.squeeze(-1)
    if gt_depth.ndim == 4:
        gt_depth = gt_depth.reshape(-1, *gt_depth.shape[-2:])

    total = 0
    for item in config:
        if item is None:
            continue
        if not isinstance(item, (dict, Mapping)):
            raise TypeError("Each gradient_loss_config entry must be a mapping")

        loss_type = item.get("type")
        if not loss_type:
            raise ValueError("Each gradient_loss_config entry requires a type")
        weight = item.get("weight", 1.0)
        if weight == 0:
            continue
        scales = item.get("scales")
        min_valid_samples = item.get("min_valid_samples")

        if loss_type == "normal":
            raw_loss = gradient_loss_multi_scale_wrapper(
                prediction,
                target,
                mask,
                gradient_loss_fn=normal_loss,
                scales=scales if scales is not None else 3,
                min_valid_samples=min_valid_samples
                if min_valid_samples is not None
                else 4096,
                gt_depth=gt_depth,
            )
        elif loss_type == "grad":
            raw_loss = gradient_loss_multi_scale_wrapper(
                prediction,
                target,
                mask,
                gradient_loss_fn=gradient_loss,
                scales=scales if scales is not None else 4,
                min_valid_samples=min_valid_samples
                if min_valid_samples is not None
                else 4096,
                scale_by_step=True,
            )
        else:
            raise ValueError(f"Unknown gradient loss type: {loss_type}")
        total += raw_loss * weight

    return total


def gradient_loss_multi_scale_wrapper(
    prediction,
    target,
    mask,
    *,
    scales,
    gradient_loss_fn,
    min_valid_samples,
    gt_depth=None,
    scale_by_step=False,
):
    """Average a geometry loss over randomly offset powers-of-two grids."""
    total = 0
    weight_sum = 0

    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        # Reduce min_valid_samples by 2 for non-zero scales, with a minimum of 16
        scale_min_valid_samples = (
            max(min_valid_samples // 2, 64) if scale > 0 else min_valid_samples
        )

        # Randomize grid offset to avoid aliasing artifacts without blurring
        if step > 1:
            offset_x = torch.randint(0, step, (1,)).item()
            offset_y = torch.randint(0, step, (1,)).item()
        else:
            offset_x = 0
            offset_y = 0

        weight = 1.0 / step if scale_by_step else 1.0

        loss_kwargs = {
            "min_valid_samples": scale_min_valid_samples,
        }
        if gt_depth is not None:
            loss_kwargs["gt_depth"] = gt_depth[:, offset_y::step, offset_x::step]

        loss_val = gradient_loss_fn(
            prediction[:, offset_y::step, offset_x::step],
            target[:, offset_y::step, offset_x::step],
            mask[:, offset_y::step, offset_x::step],
            **loss_kwargs,
        )

        total += loss_val * weight
        weight_sum += weight

    # Normalize by sum of weights to keep output scale consistent independent of number of scales
    return total / (weight_sum + 1e-6)


def gradient_loss(local_points, target_local_points, mask, min_valid_samples=1000):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.

    Args:
        local_points: (B, H, W, 3) predicted local points
        target_local_points: (B, H, W, 3) ground truth local points
        mask: (B, H, W) valid pixel mask
        min_valid_samples: Minimum number of valid edge samples required to compute loss
    """
    prediction = local_points[..., 2:3]
    target = target_local_points[..., 2:3]

    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Sum gradients and normalize by number of valid edges (not pixels)
    num = grad_x.sum() + grad_y.sum()
    denom = mask_x.sum() + mask_y.sum()

    if denom < min_valid_samples:
        return 0
    else:
        return num / denom


def _depth_edge(depth: torch.Tensor, mask: torch.Tensor) -> torch.BoolTensor:
    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    mask = mask.reshape(-1, 1, *shape[-2:])
    masked = torch.where(mask, depth, -torch.inf)
    masked_negative = torch.where(mask, -depth, -torch.inf)
    difference = F.max_pool2d(masked, 3, stride=1, padding=1) + F.max_pool2d(
        masked_negative,
        3,
        stride=1,
        padding=1,
    )
    return ((difference / depth).nan_to_num_() > 0.03).reshape(*shape)


def _smooth_huber(error: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.where(
        error < beta,
        0.5 * error.square() / beta,
        error - 0.5 * beta,
    )


def _angle_diff_vec3(vector_a: torch.Tensor, vector_b: torch.Tensor) -> torch.Tensor:
    return torch.atan2(
        torch.cross(vector_a, vector_b, dim=-1).norm(dim=-1) + 1e-12,
        (vector_a * vector_b).sum(dim=-1),
    )


def normal_loss(
    prediction,
    target,
    mask,
    min_valid_samples=1000,
    gt_depth=None,
):
    """Compute angular normal loss away from ground-truth depth edges."""
    if prediction.shape[-1] != 3:
        raise ValueError(f"Expected 3D points, got {prediction.shape[-1]} channels")

    with torch.autocast(device_type="cuda", enabled=False):
        prediction = prediction.float()
        target = target.float()

        if gt_depth is None:
            raise ValueError("gt_depth must be provided for normal_loss.")

        # Apply depth edge filtering to exclude edge regions
        not_edge = ~_depth_edge(gt_depth, mask)
        mask = mask.bool() & not_edge

        # Get 4 corners of each 2x2 patch
        # Shape: (B, H-1, W-1, 3)
        pred_leftup = prediction[:, :-1, :-1, :]
        pred_rightup = prediction[:, :-1, 1:, :]
        pred_leftdown = prediction[:, 1:, :-1, :]
        pred_rightdown = prediction[:, 1:, 1:, :]

        gt_leftup = target[:, :-1, :-1, :]
        gt_rightup = target[:, :-1, 1:, :]
        gt_leftdown = target[:, 1:, :-1, :]
        gt_rightdown = target[:, 1:, 1:, :]

        # Compute cross products for 4 different normal directions (unnormalized)
        # These are the surface normals computed from different triangle configurations
        pred_upxleft = torch.cross(
            pred_rightup - pred_rightdown, pred_leftdown - pred_rightdown, dim=-1
        )
        pred_leftxdown = torch.cross(
            pred_leftup - pred_rightup, pred_rightdown - pred_rightup, dim=-1
        )
        pred_downxright = torch.cross(
            pred_leftdown - pred_leftup, pred_rightup - pred_leftup, dim=-1
        )
        pred_rightxup = torch.cross(
            pred_rightdown - pred_leftdown, pred_leftup - pred_leftdown, dim=-1
        )

        gt_upxleft = torch.cross(
            gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1
        )
        gt_leftxdown = torch.cross(
            gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1
        )
        gt_downxright = torch.cross(
            gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1
        )
        gt_rightxup = torch.cross(
            gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1
        )

        # Compute validity masks for each corner
        # Shape: (B, H-1, W-1)
        mask_leftup = mask[:, :-1, :-1]
        mask_rightup = mask[:, :-1, 1:]
        mask_leftdown = mask[:, 1:, :-1]
        mask_rightdown = mask[:, 1:, 1:]

        mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
        mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
        mask_downxright = mask_leftdown & mask_rightup & mask_leftup
        mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

        # Count total valid samples
        total_valid = (
            mask_upxleft.sum()
            + mask_leftxdown.sum()
            + mask_downxright.sum()
            + mask_rightxup.sum()
        )

        # Early return if not enough valid samples
        if total_valid < min_valid_samples:
            return 0

        # Angular difference parameters
        MIN_ANGLE = math.radians(1)
        MAX_ANGLE = math.radians(90)
        BETA_RAD = math.radians(3.0)

        # Compute angular differences for each direction
        angle_upxleft = _smooth_huber(
            _angle_diff_vec3(pred_upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE),
            beta=BETA_RAD,
        )
        angle_leftxdown = _smooth_huber(
            _angle_diff_vec3(pred_leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE),
            beta=BETA_RAD,
        )
        angle_downxright = _smooth_huber(
            _angle_diff_vec3(pred_downxright, gt_downxright).clamp(
                MIN_ANGLE, MAX_ANGLE
            ),
            beta=BETA_RAD,
        )
        angle_rightxup = _smooth_huber(
            _angle_diff_vec3(pred_rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE),
            beta=BETA_RAD,
        )

        # Compute masked loss for each direction
        loss_upxleft = (angle_upxleft * mask_upxleft.float()).sum()
        loss_leftxdown = (angle_leftxdown * mask_leftxdown.float()).sum()
        loss_downxright = (angle_downxright * mask_downxright.float()).sum()
        loss_rightxup = (angle_rightxup * mask_rightxup.float()).sum()

        # Total loss normalized by valid samples
        total_loss = loss_upxleft + loss_leftxdown + loss_downxright + loss_rightxup

        # Check for inf/nan
        total_loss = check_and_fix_inf_nan(total_loss, "normal_loss")

        # Normalize by total valid samples
        loss = total_loss / total_valid

    return loss
