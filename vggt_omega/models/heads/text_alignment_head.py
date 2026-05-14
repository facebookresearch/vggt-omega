from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.layers import SelfAttentionBlock


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_ratio: Union[int, float, List[float]],
    *,
    eps: float,
) -> nn.Sequential:
    if isinstance(hidden_ratio, (float, int)):
        hidden_ratio = [hidden_ratio]

    layers = []
    current_dim = input_dim
    for ratio in hidden_ratio:
        hidden_dim = int(input_dim * ratio)
        layers.append(nn.Linear(current_dim, hidden_dim, bias=True))
        layers.append(nn.GELU())
        layers.append(nn.LayerNorm(hidden_dim, eps=eps))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim, bias=True))
    return nn.Sequential(*layers)


class SequenceAlignmentStudent(nn.Module):
    def __init__(
        self,
        dim_in: int = 2048,
        embedding_dim: int = 2048,
        projector_mlp_ratio: Union[int, float, List[float]] = 0.5,
        attention_depth: int = 4,
        attention_dim: int = -1,
        attention_num_heads: int = 16,
        attention_mlp_ratio: float = 4.0,
        attention_qkv_bias: bool = True,
        attention_proj_bias: bool = True,
        attention_ffn_bias: bool = True,
        attention_use_qk_norm: bool = False,
        attention_init_values: float = 1e-5,
        attention_mask_k_bias: bool = True,
        attention_post_norm: bool = True,
        attention_pre_norm: bool = True,
        disable_last_layer_amp: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        if attention_depth <= 0:
            raise ValueError("SequenceAlignmentStudent requires attention_depth > 0")

        attention_dim = dim_in if attention_dim < 0 else attention_dim
        self.disable_last_layer_amp = disable_last_layer_amp
        self.extra_attention_use_qk_norm = attention_use_qk_norm

        self.pre_norm = nn.LayerNorm(dim_in, eps=eps) if attention_pre_norm else nn.Identity()
        self.in_proj = nn.Identity() if attention_dim == dim_in else nn.Linear(dim_in, attention_dim)
        self.sequence_token = nn.Parameter(torch.zeros(1, 1, attention_dim))
        nn.init.trunc_normal_(self.sequence_token, std=0.02)

        self.extra_attention_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=attention_dim,
                    num_heads=attention_num_heads,
                    ffn_ratio=attention_mlp_ratio,
                    qkv_bias=attention_qkv_bias,
                    proj_bias=attention_proj_bias,
                    ffn_bias=attention_ffn_bias,
                    init_values=attention_init_values,
                    use_qk_norm=attention_use_qk_norm,
                    mask_k_bias=attention_mask_k_bias,
                )
                for _ in range(attention_depth)
            ]
        )
        self.token_norm = nn.LayerNorm(attention_dim, eps=eps) if attention_post_norm else nn.Identity()
        self.projector = _build_mlp(
            attention_dim,
            embedding_dim,
            projector_mlp_ratio,
            eps=eps,
        )

    def forward(self, tokens: torch.Tensor, patch_start_idx: int) -> Dict[str, torch.Tensor]:
        if patch_start_idx is None:
            raise ValueError("patch_start_idx is required for alignment student")
        if patch_start_idx > tokens.shape[2]:
            raise ValueError(f"patch_start_idx ({patch_start_idx}) exceeds token length ({tokens.shape[2]})")

        if self.disable_last_layer_amp and tokens.dtype != torch.float32:
            tokens = tokens.float()

        batch_size, num_frames, _, _ = tokens.shape
        special_tokens = tokens[:, :, :patch_start_idx, :]
        special_tokens = self.pre_norm(special_tokens)
        special_tokens = self.in_proj(special_tokens)
        special_tokens = special_tokens.reshape(batch_size, num_frames * patch_start_idx, -1)

        sequence_token = self.sequence_token.expand(batch_size, -1, -1)
        joint_tokens = torch.cat([sequence_token, special_tokens], dim=1)
        rope_sincos = None

        for block in self.extra_attention_blocks:
            joint_tokens = block(joint_tokens, rope_sincos)

        sequence_token = self.token_norm(joint_tokens[:, 0])
        projected = self.projector(sequence_token)
        return {
            "alignment_student_embedding": F.normalize(projected, dim=-1),
            "alignment_student_token": sequence_token,
        }


class TextAlignmentHead(nn.Module):
    """Student-only text alignment head for released VGGT-Omega inference."""

    def __init__(
        self,
        dim_in: int = 2048,
        embedding_dim: int = 2048,
        projector_mlp_ratio: Union[int, float, List[float]] = [0.5],
        attention_depth: int = 4,
        attention_dim: int = -1,
        attention_num_heads: int = 16,
        attention_mlp_ratio: float = 4.0,
        attention_use_qk_norm: bool = False,
        attention_pre_norm: bool = True,
        attention_post_norm: bool = True,
        disable_last_layer_amp: bool = True,
    ) -> None:
        super().__init__()
        self.student = SequenceAlignmentStudent(
            dim_in=dim_in,
            embedding_dim=embedding_dim,
            projector_mlp_ratio=projector_mlp_ratio,
            attention_depth=attention_depth,
            attention_dim=attention_dim,
            attention_num_heads=attention_num_heads,
            attention_mlp_ratio=attention_mlp_ratio,
            attention_use_qk_norm=attention_use_qk_norm,
            attention_pre_norm=attention_pre_norm,
            attention_post_norm=attention_post_norm,
            disable_last_layer_amp=disable_last_layer_amp,
        )

    @property
    def extra_attention_blocks(self):
        return self.student.extra_attention_blocks

    @property
    def extra_attention_use_qk_norm(self):
        return self.student.extra_attention_use_qk_norm

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
    ) -> Dict[str, torch.Tensor]:
        return self.student(aggregated_tokens_list[-1], patch_start_idx)
