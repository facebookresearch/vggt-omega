# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.layers import SelfAttentionBlock


class TextAlignmentHead(nn.Module):
    """Read out a language-aligned sequence embedding from camera/register tokens."""

    def __init__(self, dim_in: int = 2048) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(dim_in, eps=1e-5)

        self.language_token = nn.Parameter(torch.zeros(1, 1, dim_in))
        nn.init.trunc_normal_(self.language_token, std=0.02)

        self.readout_blocks = nn.ModuleList(
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
        self.language_token_norm = nn.LayerNorm(dim_in, eps=1e-5)
        self.embedding_projector = nn.Sequential(
            nn.Linear(dim_in, dim_in // 2, bias=True),
            nn.GELU(),
            nn.LayerNorm(dim_in // 2, eps=1e-5),
            nn.Linear(dim_in // 2, dim_in, bias=True),
        )

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        patch_token_start: int,
    ) -> dict[str, torch.Tensor]:
        tokens = aggregated_tokens_list[-1]
        if tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which TextAlignmentHead needs.")
        if patch_token_start is None:
            raise ValueError("patch_token_start is required for TextAlignmentHead")
        if patch_token_start > tokens.shape[2]:
            raise ValueError(f"patch_token_start ({patch_token_start}) exceeds token length ({tokens.shape[2]})")

        if tokens.dtype != torch.float32:
            tokens = tokens.float()

        batch_size, num_frames, _, _ = tokens.shape
        camera_and_register_tokens = tokens[:, :, :patch_token_start]
        camera_and_register_tokens = self.token_norm(camera_and_register_tokens)
        camera_and_register_tokens = camera_and_register_tokens.reshape(batch_size, num_frames * patch_token_start, -1)

        language_token = self.language_token.expand(batch_size, -1, -1)
        readout_tokens = torch.cat([language_token, camera_and_register_tokens], dim=1)
        for block in self.readout_blocks:
            readout_tokens = block(readout_tokens, None)

        language_token = self.language_token_norm(readout_tokens[:, 0])
        text_alignment_embedding = self.embedding_projector(language_token)
        return {
            "text_alignment_embedding": F.normalize(text_alignment_embedding, dim=-1),
            "text_alignment_token": language_token,
        }
