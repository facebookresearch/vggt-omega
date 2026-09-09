# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from train_utils.normalization import normalize_camera_extrinsics_and_points_batch


def normalize_gt_batch(batch: dict, normalization_type: Optional[str] = None) -> dict:
    """
    Normalize GT batch with configurable normalization types.

    Args:
        batch: Dictionary containing ground truth data.
        normalization_type: None/False to skip, or "global" for world-point scale.
    """
    if normalization_type is None or normalization_type is False:
        return batch

    if normalization_type == "global":
        batch = normalize_camera_extrinsics_and_points_batch(batch)
    else:
        raise ValueError(f"Unknown normalize_gt option: {normalization_type}")

    # Force first camera extrinsics to exact identity [I | 0] to avoid
    # numerical drift from matrix inversion / multiplication.
    # extrinsics shape: (B, S, 3, 4)
    extr = batch["extrinsics"]
    identity_3x4 = torch.eye(3, 4, device=extr.device, dtype=extr.dtype)  # (3, 4)
    if extr.ndim == 4:
        batch["extrinsics"] = torch.cat(
            [
                identity_3x4.unsqueeze(0).unsqueeze(0).expand(extr.shape[0], 1, -1, -1),
                extr[:, 1:],
            ],
            dim=1,
        )
    else:
        raise ValueError(f"Expected extrinsics (B, S, 3, 4), got {extr.shape}")

    return batch
