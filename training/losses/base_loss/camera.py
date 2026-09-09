# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from losses.metric import normalized_error_histogram, se3_to_relative_pose_error
from train_utils.general import check_and_fix_inf_nan
from vggt_omega.utils.pose_enc import encoding_to_camera, extri_intri_to_pose_encoding
from vggt_omega.utils.rotation import quat_to_mat


TRANSLATION_SLICE = slice(0, 3)
ROTATION_SLICE = slice(3, 7)
FOCAL_SLICE = slice(7, 9)


def _robust_direction_loss(
    pred_translation: torch.Tensor, gt_translation: torch.Tensor
) -> torch.Tensor:
    pred_flat = pred_translation.reshape(-1, 3)
    gt_flat = gt_translation.reshape(-1, 3)
    padding = torch.ones(
        (pred_flat.shape[0], 1),
        device=pred_flat.device,
        dtype=pred_flat.dtype,
    )
    pred_augmented = torch.cat([pred_flat, padding], dim=-1)
    gt_augmented = torch.cat([gt_flat, padding], dim=-1)
    raw_loss = (
        pred_augmented.norm(dim=-1) * gt_augmented.norm(dim=-1)
        - (pred_augmented * gt_augmented).sum(dim=-1)
    ).clamp(min=0.0)
    return torch.nan_to_num(
        torch.log1p(raw_loss),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).mean()


class PairwisePoseLoss(torch.nn.Module):
    """Compute relative rotation and translation losses for all frame pairs."""

    def __init__(self, weight_rotation=1.0, weight_translation=1.0):
        super().__init__()
        self.weight_rotation = weight_rotation
        self.weight_translation = weight_translation

    def forward(self, pred_rotation, pred_translation, gt_rotation, gt_translation):
        _, num_frames = pred_rotation.shape[:2]
        if num_frames < 2:
            return pred_rotation.sum() * 0.0

        frame_i, frame_j = torch.triu_indices(
            num_frames,
            num_frames,
            offset=1,
            device=pred_rotation.device,
        )
        pred_rotation_i = pred_rotation[:, frame_i]
        pred_rotation_j = pred_rotation[:, frame_j]
        pred_translation_i = pred_translation[:, frame_i]
        pred_translation_j = pred_translation[:, frame_j]
        gt_rotation_i = gt_rotation[:, frame_i]
        gt_rotation_j = gt_rotation[:, frame_j]
        gt_translation_i = gt_translation[:, frame_i]
        gt_translation_j = gt_translation[:, frame_j]

        pred_relative_rotation = pred_rotation_j @ pred_rotation_i.transpose(-1, -2)
        gt_relative_rotation = gt_rotation_j @ gt_rotation_i.transpose(-1, -2)
        pred_relative_translation = pred_translation_j - (
            pred_relative_rotation @ pred_translation_i.unsqueeze(-1)
        ).squeeze(-1)
        gt_relative_translation = gt_translation_j - (
            gt_relative_rotation @ gt_translation_i.unsqueeze(-1)
        ).squeeze(-1)

        rotation_loss = torch.nan_to_num(
            ((pred_relative_rotation - gt_relative_rotation) ** 2).sum(dim=(-2, -1)),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).mean()
        translation_l1 = torch.nan_to_num(
            (pred_relative_translation - gt_relative_translation).abs(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).mean()
        translation_loss = translation_l1 + 2.0 * _robust_direction_loss(
            pred_relative_translation,
            gt_relative_translation,
        )
        total_loss = (
            self.weight_rotation * rotation_loss
            + self.weight_translation * translation_loss
        )
        return total_loss


def _quaternion_l1(pred_pose, gt_pose):
    pred_quaternion = pred_pose[..., ROTATION_SLICE]
    gt_quaternion = gt_pose[..., ROTATION_SLICE]
    distance_positive = (pred_quaternion - gt_quaternion).abs().sum(dim=-1)
    distance_negative = (pred_quaternion + gt_quaternion).abs().sum(dim=-1)
    gt_target = torch.where(
        (distance_positive < distance_negative)[..., None],
        gt_quaternion,
        -gt_quaternion,
    )
    quaternion_l1 = (pred_quaternion - gt_target).abs()
    pred_normalized = F.normalize(pred_quaternion, p=2, dim=-1, eps=1e-8)
    gt_normalized = F.normalize(gt_quaternion, p=2, dim=-1, eps=1e-8)
    cosine_loss = 1.0 - (pred_normalized * gt_normalized).sum(dim=-1).abs().clamp(
        max=1.0
    )
    return quaternion_l1 + 0.1 * cosine_loss[..., None]


def camera_loss_single(
    pred_pose,
    gt_pose,
    normalize_trans_by_gt_scale,
):
    translation_loss = (
        pred_pose[..., TRANSLATION_SLICE] - gt_pose[..., TRANSLATION_SLICE]
    ).abs()
    rotation_loss = _quaternion_l1(pred_pose, gt_pose)
    focal_loss = (pred_pose[..., FOCAL_SLICE] - gt_pose[..., FOCAL_SLICE]).abs()

    if normalize_trans_by_gt_scale:
        gt_scale = (
            gt_pose[..., TRANSLATION_SLICE]
            .norm(dim=-1)
            .mean(dim=1)
            .clamp(min=0.25, max=4)
        )
        translation_loss = translation_loss / gt_scale[:, None, None]

    translation_loss = check_and_fix_inf_nan(
        translation_loss, "loss_T", hard_max=128
    ).mean()
    rotation_loss = check_and_fix_inf_nan(rotation_loss, "loss_R").mean()
    focal_loss = check_and_fix_inf_nan(focal_loss, "loss_FL").mean()
    return translation_loss, rotation_loss, focal_loss


def compute_camera_loss(
    predictions,
    batch,
    weight_trans=1.0,
    weight_rot=1.0,
    weight_focal=0.5,
    weight_pairwise=0.0,
    pairwise_loss_fn=None,
    normalize_trans_by_gt_scale=False,
):
    pred_pose_list = predictions["pose_enc_list"]
    if len(pred_pose_list) != 1:
        raise ValueError("Camera loss requires exactly one pose stage")
    pred_pose = pred_pose_list[0]
    valid_seq_mask = batch["valid_seq_mask"]
    gt_extrinsics = batch["extrinsics"]
    image_hw = batch["images"].shape[-2:]
    gt_pose = extri_intri_to_pose_encoding(
        gt_extrinsics,
        batch["intrinsics"],
        image_hw,
    )

    if valid_seq_mask.any():
        loss_translation, loss_rotation, loss_focal = camera_loss_single(
            pred_pose[valid_seq_mask],
            gt_pose[valid_seq_mask],
            normalize_trans_by_gt_scale=normalize_trans_by_gt_scale,
        )
        if weight_pairwise > 0 and pairwise_loss_fn is not None:
            pred_rotation = quat_to_mat(pred_pose[..., ROTATION_SLICE])
            pred_translation = pred_pose[..., TRANSLATION_SLICE]
            gt_rotation = gt_extrinsics[..., :3, :3]
            gt_translation = gt_extrinsics[..., :3, 3]
            pairwise_loss = pairwise_loss_fn(
                pred_rotation[valid_seq_mask].to(dtype=gt_rotation.dtype),
                pred_translation[valid_seq_mask].to(dtype=gt_translation.dtype),
                gt_rotation[valid_seq_mask],
                gt_translation[valid_seq_mask],
            )
            pairwise_loss = check_and_fix_inf_nan(pairwise_loss, "loss_PW")
        else:
            pairwise_loss = pred_pose.sum() * 0.0
    else:
        zero = pred_pose.sum() * 0.0
        loss_translation = zero
        loss_rotation = zero
        loss_focal = zero
        pairwise_loss = zero

    total_loss = (
        weight_trans * loss_translation
        + weight_rot * loss_rotation
        + weight_focal * loss_focal
        + weight_pairwise * pairwise_loss
    )

    with torch.no_grad():
        pred_extrinsics, _ = encoding_to_camera(pred_pose, image_hw)
        batch_size, num_frames = pred_extrinsics.shape[:2]
        pred_se3 = F.pad(pred_extrinsics, (0, 0, 0, 1), "constant", 0)
        pred_se3[:, :, -1, -1] = 1.0
        pred_se3 = pred_se3.reshape(batch_size * num_frames, 4, 4)
        gt_se3 = F.pad(gt_extrinsics, (0, 0, 0, 1), "constant", 0)
        gt_se3[:, :, -1, -1] = 1.0
        gt_se3 = gt_se3.reshape(batch_size * num_frames, 4, 4)
        rotation_error, translation_error = se3_to_relative_pose_error(
            pred_se3,
            gt_se3,
            num_frames,
            batch_size=batch_size,
        )
        if rotation_error.numel() == 0:
            rotation_error = pred_se3.new_zeros(1)
            translation_error = pred_se3.new_zeros(1)

        metrics = {}
        for threshold in (5, 15):
            metrics[f"Rac_{threshold}"] = (rotation_error < threshold).float().mean()
            metrics[f"Tac_{threshold}"] = (translation_error < threshold).float().mean()
        histogram = normalized_error_histogram(
            rotation_error,
            translation_error,
            max_threshold=30,
        )
        for threshold in (30, 10, 5, 3):
            metrics[f"Auc_{threshold}"] = torch.cumsum(
                histogram[:threshold], dim=0
            ).mean()
        pred_translation_max = pred_pose[..., TRANSLATION_SLICE].norm(dim=-1).max()

    loss_dict = {
        "loss_camera": total_loss,
        "loss_T": loss_translation,
        "pred_T_max": pred_translation_max,
        "loss_R": loss_rotation,
        "loss_FL": loss_focal,
        "loss_PW": pairwise_loss,
    }
    loss_dict.update(metrics)
    return loss_dict
