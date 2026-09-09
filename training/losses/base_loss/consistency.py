# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def compute_geometric_consistency_loss(
    predictions,
    batch,
    max_pixel_loss=1000.0,
    min_valid_pixels=100,
):
    """Compare predicted and ground-truth point offsets along positive tracks."""
    pred_world_points = predictions["world_points"]
    gt_world_points = batch["world_points"]
    point_masks = batch["point_masks"]
    tracks = batch["tracks"]
    track_visibility = batch["track_vis_mask"]
    batch_size, num_frames, height, width, _ = pred_world_points.shape
    num_tracks = tracks.shape[2]

    if num_tracks == 0 or num_frames <= 1:
        zero = pred_world_points.sum() * 0.0
        return {
            "loss_consistency": zero,
            "loss_consistency_max": zero,
            "consistency_valid_count": 0,
        }

    track_pixels = tracks.floor().long()
    track_pixels[..., 0].clamp_(0, width - 1)
    track_pixels[..., 1].clamp_(0, height - 1)
    x = track_pixels[..., 0]
    y = track_pixels[..., 1]

    batch_indices = torch.arange(batch_size, device=pred_world_points.device)
    reference_batch_indices = batch_indices[:, None].expand(batch_size, num_tracks)
    all_batch_indices = batch_indices[:, None, None].expand(
        batch_size,
        num_frames,
        num_tracks,
    )
    frame_indices = torch.arange(num_frames, device=pred_world_points.device)
    frame_indices = frame_indices[None, :, None].expand(
        batch_size,
        num_frames,
        num_tracks,
    )

    pred_reference = pred_world_points[:, 0][
        reference_batch_indices,
        y[:, 0],
        x[:, 0],
    ]
    gt_reference = gt_world_points[:, 0][
        reference_batch_indices,
        y[:, 0],
        x[:, 0],
    ]
    pred_all = pred_world_points[all_batch_indices, frame_indices, y, x]
    gt_all = gt_world_points[all_batch_indices, frame_indices, y, x]
    reference_point_mask = point_masks[:, 0][
        reference_batch_indices,
        y[:, 0],
        x[:, 0],
    ]
    all_point_masks = point_masks[all_batch_indices, frame_indices, y, x]

    valid_mask = (
        track_visibility[:, :1]
        & track_visibility
        & reference_point_mask[:, None]
        & all_point_masks
    )
    valid_seq_mask = batch["valid_seq_mask"] & batch["depth_train_mask"]
    valid_mask = valid_mask & valid_seq_mask[:, None, None]
    valid_mask[:, 0] = False

    valid_count = valid_mask.sum().item()
    if valid_count < min_valid_pixels:
        zero = pred_world_points.sum() * 0.0
        return {
            "loss_consistency": zero,
            "loss_consistency_max": zero,
            "consistency_valid_count": valid_count,
        }

    pred_difference = pred_reference[:, None] - pred_all
    gt_difference = gt_reference[:, None] - gt_all
    valid_errors = (pred_difference - gt_difference).abs()[
        valid_mask[..., None].expand(-1, -1, -1, 3)
    ]
    if max_pixel_loss is not None:
        valid_errors = valid_errors.clamp(max=max_pixel_loss)
    return {
        "loss_consistency": valid_errors.mean(),
        "loss_consistency_max": valid_errors.max(),
        "consistency_valid_count": valid_count,
    }
