import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.layers import SelfAttentionBlock


class SequenceAlignmentStudent(nn.Module):
    """Student-only sequence embedding head used by the aligned checkpoint."""

    def __init__(self, dim_in: int = 2048) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(dim_in, eps=1e-5)
        self.input_proj = nn.Identity()
        self.sequence_token = nn.Parameter(torch.zeros(1, 1, dim_in))
        nn.init.trunc_normal_(self.sequence_token, std=0.02)

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
        self.projector = nn.Sequential(
            nn.Linear(dim_in, dim_in // 2, bias=True),
            nn.GELU(),
            nn.LayerNorm(dim_in // 2, eps=1e-5),
            nn.Linear(dim_in // 2, dim_in, bias=True),
        )

    def forward(self, tokens: torch.Tensor, patch_start_idx: int) -> dict[str, torch.Tensor]:
        if patch_start_idx is None:
            raise ValueError("patch_start_idx is required for alignment student")
        if patch_start_idx > tokens.shape[2]:
            raise ValueError(f"patch_start_idx ({patch_start_idx}) exceeds token length ({tokens.shape[2]})")

        if tokens.dtype != torch.float32:
            tokens = tokens.float()

        batch_size, num_frames, _, _ = tokens.shape
        special_tokens = tokens[:, :, :patch_start_idx]
        special_tokens = self.token_norm(special_tokens)
        special_tokens = self.input_proj(special_tokens)
        special_tokens = special_tokens.reshape(batch_size, num_frames * patch_start_idx, -1)

        sequence_token = self.sequence_token.expand(batch_size, -1, -1)
        joint_tokens = torch.cat([sequence_token, special_tokens], dim=1)
        rope_sincos = None
        for block in self.trunk:
            joint_tokens = block(joint_tokens, rope_sincos)

        sequence_token = self.trunk_norm(joint_tokens[:, 0])
        projected = self.projector(sequence_token)
        return {
            "alignment_student_embedding": F.normalize(projected, dim=-1),
            "alignment_student_token": sequence_token,
        }


class TextAlignmentHead(nn.Module):
    """Student-only text alignment head for released VGGT-Omega inference."""

    def __init__(self, dim_in: int = 2048) -> None:
        super().__init__()
        self.student = SequenceAlignmentStudent(dim_in=dim_in)

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor],
        patch_start_idx: int,
    ) -> dict[str, torch.Tensor]:
        return self.student(aggregated_tokens_list[-1], patch_start_idx)
