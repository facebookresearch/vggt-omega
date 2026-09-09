# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from losses.base_loss.depth import weighted_mean
from losses.unprojection import unproject_depth_to_points_torch_batch
from vggt_omega.utils.pose_enc import encoding_to_camera


def compute_point_loss(
    predictions,
    batch,
    intrinsics_warmup_ratio=0.5,
    schedule_progress=0.0,
    min_valid_pts=1000,
    abs_l1_loss_ratio=0.0,
    rel_l1_loss_ratio=1.0,
    max_pixel_loss=100.0,
):
    """Compute L1 loss on world points derived from depth and camera predictions."""
    pred_depth = predictions["depth"]
    if pred_depth.ndim == 4:
        pred_depth = pred_depth.unsqueeze(-1)
    pred_extrinsics, pred_intrinsics = encoding_to_camera(
        predictions["pose_enc_list"][-1],
        pred_depth.shape[-3:-1],
    )
    gt_intrinsics = batch["intrinsics"]
    if schedule_progress < intrinsics_warmup_ratio:
        final_intrinsics = gt_intrinsics.clone()
    else:
        final_intrinsics = torch.stack(
            [pred_intrinsics[..., 0], pred_intrinsics[..., 1], gt_intrinsics[..., 2]],
            dim=-1,
        )

    pred_points = unproject_depth_to_points_torch_batch(
        pred_depth,
        pred_extrinsics,
        final_intrinsics,
    )
    predictions["world_points"] = pred_points.float()

    gt_depth = batch["depths"]
    if gt_depth.ndim == 4:
        gt_depth = gt_depth.unsqueeze(-1)
    gt_points = unproject_depth_to_points_torch_batch(
        gt_depth,
        batch["extrinsics"],
        gt_intrinsics,
    )
    batch["world_points"] = gt_points

    valid_masks = batch["point_masks"]
    valid_seq_mask = batch["valid_seq_mask"] & batch["depth_train_mask"]
    if not valid_seq_mask.any():
        zero = pred_points.sum() * 0.0
        return (
            predictions,
            batch,
            {
                "loss_point": zero,
                "loss_point_l1": zero,
                "point_l1_unweighted": zero,
                "loss_point_l1_max": zero.detach(),
            },
        )

    pred_points = pred_points[valid_seq_mask]
    gt_points = gt_points[valid_seq_mask]
    valid_masks = valid_masks[valid_seq_mask]
    gt_depth = gt_depth[valid_seq_mask]
    valid_pixels = valid_masks.bool()
    if valid_pixels.any():
        point_l1_unweighted = (pred_points - gt_points).abs()[valid_pixels].mean()
    else:
        point_l1_unweighted = pred_points.sum() * 0.0

    difference = (pred_points - gt_points).abs()
    if rel_l1_loss_ratio == 0:
        weighted_difference = difference * abs_l1_loss_ratio
    else:
        depth_values = gt_depth.squeeze(-1)
        depth_mean = weighted_mean(
            depth_values,
            valid_masks,
            dim=(-2, -1),
            keepdim=True,
        )
        relative_weights = depth_values.clamp_min(0.1 * depth_mean)
        relative_weights = (1.0 / (relative_weights + 1e-6)).clamp(
            min=0.1,
            max=10.0,
        )
        weighted_difference = (
            difference * relative_weights.unsqueeze(-1) * rel_l1_loss_ratio
            + difference * abs_l1_loss_ratio
        )

    if valid_masks.sum() > min_valid_pts:
        point_l1_values = weighted_difference[valid_pixels].clamp(max=max_pixel_loss)
        point_l1 = point_l1_values.mean()
        point_l1_max = point_l1_values.max()
    else:
        point_l1 = pred_points.sum() * 0.0
        point_l1_max = pred_points.new_zeros(())

    return (
        predictions,
        batch,
        {
            "loss_point": point_l1,
            "loss_point_l1": point_l1,
            "point_l1_unweighted": point_l1_unweighted,
            "loss_point_l1_max": point_l1_max,
        },
    )
