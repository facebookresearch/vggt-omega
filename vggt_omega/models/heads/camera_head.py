# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.layers import SelfAttentionBlock


class CameraHead(nn.Module):
    """Camera head used by the released VGGT-Omega checkpoints."""

    def __init__(self, dim_in: int = 2048) -> None:
        super().__init__()

        self.token_norm = nn.LayerNorm(dim_in, eps=1e-5)
        self.input_proj = nn.Identity()
        # Head-local transformer blocks that mix special tokens across frames.
        self.trunk = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=dim_in,
                    num_heads=16,
                    ffn_ratio=4.0,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    init_values=1e-5,
                    use_qk_norm=False,
                    mask_k_bias=True,
                )
                for _ in range(4)
            ]
        )
        self.trunk_norm = nn.LayerNorm(dim_in, eps=1e-5)
        self.pose_branch = nn.Sequential(
            nn.Linear(dim_in, dim_in // 2, bias=True),
            nn.GELU(),
            nn.Linear(dim_in // 2, 9, bias=True),
        )

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor],
        patch_start_idx: int,
    ) -> torch.Tensor:
        tokens = aggregated_tokens_list[-1]
        batch_size, num_frames, num_tokens, _ = tokens.shape

        if patch_start_idx is None:
            raise ValueError("patch_start_idx is required for CameraHead")
        if patch_start_idx > num_tokens:
            raise ValueError(f"patch_start_idx ({patch_start_idx}) exceeds token length ({num_tokens})")

        head_context = torch.autocast(device_type="cuda", enabled=False) if tokens.is_cuda else contextlib.nullcontext()
        with head_context:
            if tokens.dtype != torch.float32:
                tokens = tokens.float()

            special_tokens = tokens[:, :, :patch_start_idx]
            special_tokens = self.token_norm(special_tokens)
            special_tokens = self.input_proj(special_tokens)

            special_tokens = special_tokens.reshape(batch_size, num_frames * patch_start_idx, -1)
            rope_sincos = None
            for block in self.trunk:
                special_tokens = block(special_tokens, rope_sincos)

            special_tokens = special_tokens.reshape(batch_size, num_frames, patch_start_idx, -1)
            pose_tokens = self.trunk_norm(special_tokens[:, :, 0])
            return _apply_pose_activation(self.pose_branch(pose_tokens))


def _apply_pose_activation(pred_pose_enc: torch.Tensor) -> torch.Tensor:
    translation = pred_pose_enc[..., :3]
    quaternion = pred_pose_enc[..., 3:7]
    fov = F.relu(pred_pose_enc[..., 7:]) + 0.01
    return torch.cat([translation, quaternion, fov], dim=-1)
