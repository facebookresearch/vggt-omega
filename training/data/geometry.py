# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def project_world_points_to_cam(
    world_points,
    cam_extrinsics,
    cam_intrinsics,
    default=0,
):
    """
    Transforms 3D points to 2D using extrinsic and intrinsic parameters.
    Args:
        world_points (torch.Tensor): 3D points of shape Px3.
        cam_extrinsics (torch.Tensor): Extrinsic parameters of shape Bx3x4.
        cam_intrinsics (torch.Tensor): Intrinsic parameters of shape Bx3x3.
    Returns:
        torch.Tensor: Transformed 2D points of shape BxNx2.
    """
    device = world_points.device
    with torch.autocast(device_type=device.type, enabled=False):
        B = cam_extrinsics.shape[0]  # Batch size, i.e., number of cameras
        world_points_homogeneous = torch.cat(
            [world_points, torch.ones_like(world_points[..., 0:1])], dim=1
        )  # Nx4
        # Reshape for batch processing
        world_points_homogeneous = world_points_homogeneous.unsqueeze(0).expand(
            B, -1, -1
        )  # BxNx4

        # Step 1: Apply extrinsic parameters
        # Transform 3D points to camera coordinate system for all cameras
        cam_points = torch.bmm(
            cam_extrinsics, world_points_homogeneous.transpose(-1, -2)
        )

        image_points = img_from_cam(cam_intrinsics, cam_points, default=default)

        return image_points, cam_points


def img_from_cam(cam_intrinsics, cam_points, default=0.0):
    """
    Applies intrinsic parameters to the given 3D points.

    Args:
        cam_intrinsics (torch.Tensor): Intrinsic camera parameters of shape Bx3x3.
        cam_points (torch.Tensor): 3D points in camera coordinates of shape Bx3xN.
        default (float, optional): Default value to replace NaNs in the output.

    Returns:
        pixel_coords (torch.Tensor): 2D points in pixel coordinates of shape BxNx2.
    """

    # Normalized device coordinates (NDC)
    cam_points = cam_points / cam_points[:, 2:3, :]
    ndc_xy = cam_points[:, :2, :]

    # Prepare cam_points for batch matrix multiplication
    cam_coords_homo = torch.cat(
        (ndc_xy, torch.ones_like(ndc_xy[:, :1, :])), dim=1
    )  # Bx3xN
    # Apply intrinsic parameters using batch matrix multiplication
    pixel_coords = torch.bmm(cam_intrinsics, cam_coords_homo)  # Bx3xN

    # Extract x and y coordinates
    pixel_coords = pixel_coords[:, :2, :]  # Bx2xN

    # Replace NaNs with default value
    pixel_coords = torch.nan_to_num(pixel_coords, nan=default)

    return pixel_coords.transpose(1, 2)  # BxNx2
