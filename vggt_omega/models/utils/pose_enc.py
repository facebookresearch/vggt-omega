# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .rotation import quat_to_mat, mat_to_quat
from .geometry import closed_form_inverse_se3


def extri_intri_to_pose_encoding(
    extrinsics, intrinsics, image_size_hw=None, pose_encoding_type="absT_quaR_FoV", pixel_convention: str = "colmap"  # e.g., (256, 512)
):
    """Convert camera extrinsics and intrinsics to a compact pose encoding.

    This function transforms camera parameters into a unified pose encoding format,
    which can be used for various downstream tasks like pose prediction or representation.

    Args:
        extrinsics (torch.Tensor): Camera extrinsic parameters with shape BxSx3x4,
            where B is batch size and S is sequence length.
            In OpenCV coordinate system (x-right, y-down, z-forward), representing
            camera from world (w2c) transformation for all encoding types.
            The format is [R|t] where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
        intrinsics (torch.Tensor): Camera intrinsic parameters with shape BxSx3x3.
            Defined in pixels, with format:
            [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]]
            where fx, fy are focal lengths and (cx, cy) is the principal point
        image_size_hw (tuple): Tuple of (height, width) of the image in pixels.
            Required for computing field of view values. For example: (256, 512).
        pose_encoding_type (str): Type of pose encoding to use. Supported types:
            - "absT_quaR_FoV": absolute translation, quaternion rotation, field of view.
              Input/output extrinsics are camera from world (w2c). Encoding stores w2c params.
            - "absT_quaR_FoV_PP": same as above but also encodes principal point offset.
            - "absT_quaR_FoV_c2w": input extrinsics are camera from world (w2c), but the
              encoding internally represents camera to world (c2w) params. This allows the
              model to learn c2w representation while keeping input/output as w2c.

    Returns:
        torch.Tensor: Encoded camera pose parameters with shape BxSx9.
            For "absT_quaR_FoV" and "absT_quaR_FoV_c2w" types, the 9 dimensions are:
            - [:3] = absolute translation vector T (3D)
            - [3:7] = rotation as quaternion quat (4D)
            - [7:] = field of view (2D)
    """

    # extrinsics: BxSx3x4
    # intrinsics: BxSx3x3

    if pose_encoding_type == "absT_quaR_FoV":
        R = extrinsics[:, :, :3, :3]  # BxSx3x3
        T = extrinsics[:, :, :3, 3]  # BxSx3

        quat = mat_to_quat(R)
        # Note the order of h and w here
        H, W = image_size_hw
        fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
        fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
        pose_encoding = torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()
    elif pose_encoding_type == "absT_quaR_FoV_PP":
        R = extrinsics[:, :, :3, :3]  # BxSx3x3
        T = extrinsics[:, :, :3, 3]  # BxSx3

        quat = mat_to_quat(R)
        # Note the order of h and w here
        H, W = image_size_hw
        fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
        fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
        
        if pixel_convention.lower() == "opencv":
            center_x = (W - 1) / 2.0
            center_y = (H - 1) / 2.0
        else:
            center_x = W / 2.0
            center_y = H / 2.0
        offset_x = (intrinsics[..., 0, 2] - center_x) / center_x
        offset_y = (intrinsics[..., 1, 2] - center_y) / center_y
        offset_x = offset_x * 100 # for stability
        offset_y = offset_y * 100
        
        # Normalize the offset by half the image dimensions
        # scale_xy has shape [2] which will broadcast correctly
        # scale_xy = torch.tensor([W / 2.0, H / 2.0], device=pp_xy.device, dtype=pp_xy.dtype)
        # normalized_pp = offset_xy / scale_xy
        pose_encoding = torch.cat([T, quat, fov_h[..., None], fov_w[..., None], offset_x[..., None], offset_y[..., None]], dim=-1).float()
    elif pose_encoding_type == "absT_quaR_FoV_c2w":
        # Input extrinsics is cam_from_world (w2c), but we encode in cam_to_world (c2w) representation
        # This allows the model to learn c2w pose encoding
        B, S = extrinsics.shape[:2]
        extrinsics_flat = extrinsics.reshape(B * S, 3, 4)  # (B*S, 3, 4)
        extrinsics_c2w = closed_form_inverse_se3(extrinsics_flat)[:, :3, :]  # (B*S, 3, 4)
        extrinsics_c2w = extrinsics_c2w.reshape(B, S, 3, 4)  # (B, S, 3, 4)
        
        R = extrinsics_c2w[:, :, :3, :3]  # BxSx3x3, R_c2w
        T = extrinsics_c2w[:, :, :3, 3]  # BxSx3, camera position in world

        quat = mat_to_quat(R)
        # Note the order of h and w here
        H, W = image_size_hw
        fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
        fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
        pose_encoding = torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()
    else:
        raise NotImplementedError

    return pose_encoding


def pose_encoding_to_extri_intri(
    pose_encoding, image_size_hw=None, pose_encoding_type="absT_quaR_FoV", build_intrinsics=True, pixel_convention: str = "colmap"  # e.g., (256, 512)
):
    """Convert a pose encoding back to camera extrinsics and intrinsics.

    This function performs the inverse operation of extri_intri_to_pose_encoding,
    reconstructing the full camera parameters from the compact encoding.

    Args:
        pose_encoding (torch.Tensor): Encoded camera pose parameters with shape BxSx9,
            where B is batch size and S is sequence length.
            For "absT_quaR_FoV" and "absT_quaR_FoV_c2w" types, the 9 dimensions are:
            - [:3] = absolute translation vector T (3D)
            - [3:7] = rotation as quaternion quat (4D)
            - [7:] = field of view (2D)
        image_size_hw (tuple): Tuple of (height, width) of the image in pixels.
            Required for reconstructing intrinsics from field of view values.
            For example: (256, 512).
        pose_encoding_type (str): Type of pose encoding used. Supported types:
            - "absT_quaR_FoV": outputs camera from world (w2c) extrinsics.
            - "absT_quaR_FoV_PP": same as above but also decodes principal point offset.
            - "absT_quaR_FoV_c2w": encoding internally represents c2w params, but outputs
              camera from world (w2c) extrinsics.
        build_intrinsics (bool): Whether to reconstruct the intrinsics matrix.
            If False, only extrinsics are returned and intrinsics will be None.

    Returns:
        tuple: (extrinsics, intrinsics)
            - extrinsics (torch.Tensor): Camera extrinsic parameters with shape BxSx3x4.
              In OpenCV coordinate system (x-right, y-down, z-forward), representing
              camera from world (w2c) transformation for all encoding types.
              The format is [R|t] where R is a 3x3 rotation matrix and t is
              a 3x1 translation vector.
            - intrinsics (torch.Tensor or None): Camera intrinsic parameters with shape BxSx3x3,
              or None if build_intrinsics is False. Defined in pixels, with format:
              [[fx, 0, cx],
               [0, fy, cy],
               [0,  0,  1]]
              where fx, fy are focal lengths and (cx, cy) is the principal point,
              assumed to be at the center of the image (W/2, H/2).
    """

    intrinsics = None

    if pose_encoding_type == "absT_quaR_FoV":
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]
        fov_h = pose_encoding[..., 7]
        fov_w = pose_encoding[..., 8]

        R = quat_to_mat(quat)
        extrinsics = torch.cat([R, T[..., None]], dim=-1)

        if build_intrinsics:
            H, W = image_size_hw
            fy = (H / 2.0) / torch.tan(fov_h / 2.0)
            fx = (W / 2.0) / torch.tan(fov_w / 2.0)
            intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
            intrinsics[..., 0, 0] = fx
            intrinsics[..., 1, 1] = fy
            if pixel_convention.lower() == "opencv":
                intrinsics[..., 0, 2] = (W - 1) / 2
                intrinsics[..., 1, 2] = (H - 1) / 2
            else:
                intrinsics[..., 0, 2] = W / 2
                intrinsics[..., 1, 2] = H / 2
            intrinsics[..., 2, 2] = 1.0  # Set the homogeneous coordinate to 1
            # zeros = torch.zeros_like(fx)
            # ones = torch.ones_like(fx)
            # cx = torch.full_like(fx, W / 2.0)
            # cy = torch.full_like(fx, H / 2.0)
            # row0 = torch.stack([fx, zeros, cx], dim=-1)
            # row1 = torch.stack([zeros, fy, cy], dim=-1)
            # row2 = torch.stack([zeros, zeros, ones], dim=-1)
            # intrinsics = torch.stack([row0, row1, row2], dim=-2)
    elif pose_encoding_type == "absT_quaR_FoV_PP":
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]
        fov_h = pose_encoding[..., 7]
        fov_w = pose_encoding[..., 8]
        offset_x = pose_encoding[..., 9]
        offset_y = pose_encoding[..., 10]

        R = quat_to_mat(quat)
        extrinsics = torch.cat([R, T[..., None]], dim=-1)

        if build_intrinsics:
            H, W = image_size_hw
            fy = (H / 2.0) / torch.tan(fov_h / 2.0)
            fx = (W / 2.0) / torch.tan(fov_w / 2.0)
            zeros = torch.zeros_like(fx)
            ones = torch.ones_like(fx)
            if pixel_convention.lower() == "opencv":
                center_x = torch.full_like(fx, (W - 1) / 2.0)
                center_y = torch.full_like(fx, (H - 1) / 2.0)
            else:
                center_x = torch.full_like(fx, W / 2.0)
                center_y = torch.full_like(fx, H / 2.0)
            cx = center_x * (1.0 + offset_x / 100.0)
            cy = center_y * (1.0 + offset_y / 100.0)
            row0 = torch.stack([fx, zeros, cx], dim=-1)
            row1 = torch.stack([zeros, fy, cy], dim=-1)
            row2 = torch.stack([zeros, zeros, ones], dim=-1)
            intrinsics = torch.stack([row0, row1, row2], dim=-2)
    elif pose_encoding_type == "absT_quaR_FoV_c2w":
        # Pose encoding is in c2w representation, decode and convert back to w2c
        T = pose_encoding[..., :3]  # camera position in world
        quat = pose_encoding[..., 3:7]
        fov_h = pose_encoding[..., 7]
        fov_w = pose_encoding[..., 8]

        R = quat_to_mat(quat)  # R_c2w
        extrinsics_c2w = torch.cat([R, T[..., None]], dim=-1)  # BxSx3x4
        
        # Invert from cam_to_world to cam_from_world
        B, S = extrinsics_c2w.shape[:2]
        extrinsics_flat = extrinsics_c2w.reshape(B * S, 3, 4)  # (B*S, 3, 4)
        extrinsics_w2c = closed_form_inverse_se3(extrinsics_flat)[:, :3, :]  # (B*S, 3, 4)
        extrinsics = extrinsics_w2c.reshape(B, S, 3, 4)  # (B, S, 3, 4)

        if build_intrinsics:
            H, W = image_size_hw
            fy = (H / 2.0) / torch.tan(fov_h / 2.0)
            fx = (W / 2.0) / torch.tan(fov_w / 2.0)
            intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
            intrinsics[..., 0, 0] = fx
            intrinsics[..., 1, 1] = fy
            if pixel_convention.lower() == "opencv":
                intrinsics[..., 0, 2] = (W - 1) / 2
                intrinsics[..., 1, 2] = (H - 1) / 2
            else:
                intrinsics[..., 0, 2] = W / 2
                intrinsics[..., 1, 2] = H / 2
            intrinsics[..., 2, 2] = 1.0
    else:
        raise NotImplementedError

    return extrinsics, intrinsics
