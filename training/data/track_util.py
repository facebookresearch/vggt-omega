# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from data.geometry import project_world_points_to_cam


def build_tracks_by_depth(
    extrinsics,
    intrinsics,
    world_points,
    depths,
    point_masks,
    pos_rel_thres=0.05,
    boundary_thres=4,
    target_track_num=512,
    seq_name=None,
):
    """Build fixed-size positive point tracks from the first frame's depth."""
    num_frames, height, width, _ = world_points.shape
    final_tracks = torch.zeros(
        num_frames,
        target_track_num,
        2,
        device=world_points.device,
        dtype=torch.float32,
    )
    final_vis_masks = torch.zeros(
        num_frames,
        target_track_num,
        device=world_points.device,
        dtype=torch.bool,
    )

    query_point_masks = point_masks[0]
    if not query_point_masks.any():
        logging.warning("No valid query points in %s", seq_name)
        return final_tracks, final_vis_masks

    query_world_points = world_points[0][query_point_masks]
    image_points, cam_points = project_world_points_to_cam(
        query_world_points,
        extrinsics,
        intrinsics,
    )
    projected_depths = cam_points[:, -1]

    pixel_indices = image_points.floor().long()
    inside = (
        (pixel_indices[..., 0] >= boundary_thres)
        & (pixel_indices[..., 0] < width - boundary_thres)
        & (pixel_indices[..., 1] >= boundary_thres)
        & (pixel_indices[..., 1] < height - boundary_thres)
    )
    safe_pixel_indices = pixel_indices.clone()
    safe_pixel_indices[~inside] = 0
    batch_indices = (
        torch.arange(num_frames, device=world_points.device)
        .view(num_frames, 1)
        .expand(-1, safe_pixel_indices.shape[1])
    )

    depth_visible = torch.zeros_like(inside)
    for shift in ((0, 0), (1, 0), (0, 1), (1, 1)):
        shifted_indices = safe_pixel_indices + torch.tensor(
            shift,
            device=safe_pixel_indices.device,
        )
        depth_visible |= _get_depth_inside_flag(
            depths,
            batch_indices,
            shifted_indices,
            projected_depths,
            pos_rel_thres,
        )

    sampled_tracks, sampled_vis_masks = _sample_positive_tracks(
        image_points,
        inside & depth_visible,
        target_track_num,
    )
    sampled_track_num = sampled_tracks.shape[1]
    final_tracks[:, :sampled_track_num] = sampled_tracks
    final_vis_masks[:, :sampled_track_num] = sampled_vis_masks
    return final_tracks, final_vis_masks


def _get_depth_inside_flag(
    depths, batch_indices, pixel_indices, projected_depths, rel_thres
):
    sampled_depths = depths[
        batch_indices,
        pixel_indices[..., 1],
        pixel_indices[..., 0],
    ]
    valid_depths = sampled_depths > 1e-8
    depth_diff = (projected_depths - sampled_depths).abs()
    depth_matches = (depth_diff < projected_depths * rel_thres + 0.01) & (
        depth_diff < sampled_depths * rel_thres + 0.01
    )
    return depth_matches & valid_depths


def _sample_positive_tracks(tracks, tracks_mask, track_num):
    tracks_mask = tracks_mask.clone()
    tracks_mask[:, ~tracks_mask[0]] = False

    visible_frame_counts = tracks_mask.sum(dim=0)
    tracks_mask[:, visible_frame_counts <= 1] = False
    visible_frame_counts = tracks_mask.sum(dim=0)
    sorted_indices = visible_frame_counts.argsort(descending=True)

    if len(sorted_indices) // 2 > track_num:
        sorted_indices = sorted_indices[: len(sorted_indices) // 2]

    pick_indices = torch.randperm(len(sorted_indices), device=sorted_indices.device)[
        :track_num
    ]
    selected = sorted_indices[pick_indices]
    return tracks[:, selected].clone(), tracks_mask[:, selected].clone()
