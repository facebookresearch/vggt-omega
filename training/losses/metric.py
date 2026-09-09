# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from vggt_omega.utils.geometry import closed_form_inverse_se3
from vggt_omega.utils.rotation import mat_to_quat


def build_pair_index(num_frames, batch_size):
    frame_i, frame_j = torch.combinations(
        torch.arange(num_frames),
        2,
        with_replacement=False,
    ).unbind(-1)
    offsets = torch.arange(batch_size)[:, None] * num_frames
    return (frame_i[None] + offsets).reshape(-1), (frame_j[None] + offsets).reshape(-1)


def rotation_angle(gt_rotation, pred_rotation):
    pred_quaternion = mat_to_quat(pred_rotation)
    gt_quaternion = mat_to_quat(gt_rotation)
    quaternion_loss = (1 - (pred_quaternion * gt_quaternion).sum(dim=1) ** 2).clamp(
        min=1e-15
    )
    return torch.arccos(1 - 2 * quaternion_loss) * 180 / np.pi


def translation_angle(gt_translation, pred_translation):
    error = compare_translation_by_angle(gt_translation, pred_translation)
    error = error * 180.0 / np.pi
    return torch.minimum(error, (180 - error).abs())


def compare_translation_by_angle(gt_translation, pred_translation):
    pred_translation = pred_translation / (
        pred_translation.norm(dim=1, keepdim=True) + 1e-15
    )
    gt_translation = gt_translation / (gt_translation.norm(dim=1, keepdim=True) + 1e-15)
    translation_loss = torch.clamp_min(
        1.0 - torch.sum(pred_translation * gt_translation, dim=1) ** 2,
        1e-15,
    )
    error = torch.acos(torch.sqrt(1 - translation_loss))
    error[torch.isnan(error) | torch.isinf(error)] = 1e6
    return error


def normalized_error_histogram(rotation_error, translation_error, max_threshold=30):
    max_errors = torch.maximum(rotation_error, translation_error)
    histogram = torch.histc(
        max_errors,
        bins=max_threshold + 1,
        min=0,
        max=max_threshold,
    )
    return histogram / float(max_errors.shape[0])


def se3_to_relative_pose_error(
    pred_se3,
    gt_se3,
    num_frames,
    batch_size,
):
    frame_i, frame_j = build_pair_index(num_frames, batch_size)
    gt_relative_pose = gt_se3[frame_i].bmm(closed_form_inverse_se3(gt_se3[frame_j]))
    pred_relative_pose = pred_se3[frame_i].bmm(
        closed_form_inverse_se3(pred_se3[frame_j])
    )
    rotation_error = rotation_angle(
        gt_relative_pose[:, :3, :3],
        pred_relative_pose[:, :3, :3],
    )
    translation_error = translation_angle(
        gt_relative_pose[:, :3, 3],
        pred_relative_pose[:, :3, 3],
    )
    return rotation_error, translation_error
