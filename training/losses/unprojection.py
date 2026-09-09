# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from vggt_omega.utils.geometry import closed_form_inverse_se3


def unproject_depth_to_cam_coords(depth, intrinsics):
    """Unproject COLMAP-convention depth maps into camera coordinates."""
    if depth.ndim == 4:
        depth = depth.unsqueeze(-1)
    if depth.ndim != 5:
        raise ValueError("depth must have shape (B, S, H, W[, 1])")

    batch_size, num_frames, height, width, _ = depth.shape
    intrinsics = intrinsics.to(device=depth.device, dtype=depth.dtype)
    u = (
        torch.arange(width, device=depth.device, dtype=depth.dtype).view(1, 1, 1, width)
        + 0.5
    )
    v = (
        torch.arange(height, device=depth.device, dtype=depth.dtype).view(
            1, 1, height, 1
        )
        + 0.5
    )

    focal_x = intrinsics[..., 0, 0].view(batch_size, num_frames, 1, 1)
    focal_y = intrinsics[..., 1, 1].view(batch_size, num_frames, 1, 1)
    center_x = intrinsics[..., 0, 2].view(batch_size, num_frames, 1, 1)
    center_y = intrinsics[..., 1, 2].view(batch_size, num_frames, 1, 1)
    eps = torch.finfo(depth.dtype).eps
    focal_x = focal_x.clamp(min=eps)
    focal_y = focal_y.clamp(min=eps)

    z = depth.squeeze(-1)
    x = (u - center_x) * z / focal_x
    y = (v - center_y) * z / focal_y
    return torch.stack((x, y, z), dim=-1)


def unproject_depth_to_points_torch_batch(depth, extrinsics, intrinsics):
    """Unproject depth maps into world coordinates."""
    camera_points = unproject_depth_to_cam_coords(depth, intrinsics)
    batch_size, num_frames = camera_points.shape[:2]
    extrinsics = extrinsics.to(device=camera_points.device, dtype=camera_points.dtype)
    camera_to_world = closed_form_inverse_se3(extrinsics.reshape(-1, 3, 4))
    rotation = camera_to_world[:, :3, :3].reshape(batch_size, num_frames, 3, 3)
    translation = camera_to_world[:, :3, 3].reshape(batch_size, num_frames, 3)
    return (
        torch.einsum("bshwc,bscd->bshwd", camera_points, rotation.transpose(-1, -2))
        + translation[:, :, None, None, :]
    )
