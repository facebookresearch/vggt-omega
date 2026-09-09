# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from losses.unprojection import unproject_depth_to_points_torch_batch
from train_utils.normalization import _normalize_scene_differentiable
from vggt_omega.utils.pose_enc import encoding_to_camera


def normalize_predictions_differentiable(
    predictions,
    batch,
    schedule_progress,
    rel_to_first_cam=False,
    intrinsics_warmup_ratio=0.1,
):
    """Normalize predicted camera translation and depth by the predicted scene scale."""
    pred_depth = predictions["depth"]
    pred_pose_encodings = predictions["pose_enc_list"]
    if len(pred_pose_encodings) != 1:
        raise ValueError("Prediction normalization requires exactly one pose stage")

    pred_pose = pred_pose_encodings[0]
    pred_extrinsics, pred_intrinsics = encoding_to_camera(
        pred_pose,
        batch["images"].shape[-2:],
    )
    gt_intrinsics = batch["intrinsics"]
    if schedule_progress < intrinsics_warmup_ratio:
        final_intrinsics = gt_intrinsics.clone()
    else:
        final_intrinsics = torch.stack(
            [pred_intrinsics[..., 0], pred_intrinsics[..., 1], gt_intrinsics[..., 2]],
            dim=-1,
        )

    pred_world_points = unproject_depth_to_points_torch_batch(
        pred_depth,
        pred_extrinsics,
        final_intrinsics,
    )
    avg_scale, valid_seq_mask = _normalize_scene_differentiable(
        pred_extrinsics,
        pred_world_points,
        pred_depth,
        batch["point_masks"],
        scale_only=True,
        rel_to_first_cam=rel_to_first_cam,
    )
    avg_scale = avg_scale.float().clamp(min=1e-3)

    avg_scale = torch.where(
        batch["depth_train_mask"],
        avg_scale,
        avg_scale.detach(),
    )

    batch["valid_seq_mask"] = valid_seq_mask & batch["valid_seq_mask"]
    normalized_pose = torch.cat(
        [pred_pose[..., :3] / avg_scale.view(-1, 1, 1), pred_pose[..., 3:]],
        dim=-1,
    )
    dense_scale = avg_scale.view(-1, 1, 1, 1, 1)
    predictions["pose_enc_list"] = [normalized_pose]
    predictions["depth"] = pred_depth / dense_scale
    predictions["world_points"] = (pred_world_points / dense_scale).float()
    return predictions, batch
