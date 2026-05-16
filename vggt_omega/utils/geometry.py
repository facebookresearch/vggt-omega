# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch


def closed_form_inverse_se3(se3, R=None, T=None):
    """Invert a batch of 3x4 or 4x4 SE(3) matrices."""
    is_numpy = isinstance(se3, np.ndarray)

    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must have shape (N, 4, 4) or (N, 3, 4), got {se3.shape}")

    if R is None:
        R = se3[:, :3, :3]
    if T is None:
        T = se3[:, :3, 3:]

    if is_numpy:
        R_t = np.transpose(R, (0, 2, 1))
        top_right = -np.matmul(R_t, T)
        inverted = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_t = R.transpose(1, 2)
        top_right = -torch.bmm(R_t, T)
        inverted = torch.eye(4, device=R.device, dtype=R.dtype)[None].repeat(len(R), 1, 1)

    inverted[:, :3, :3] = R_t
    inverted[:, :3, 3:] = top_right
    return inverted
