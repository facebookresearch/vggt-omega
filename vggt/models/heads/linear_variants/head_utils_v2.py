from typing import Optional

import torch
import torch.nn as nn


def build_prediction_head(
    in_channels: int,
    out_channels: int,
    proj_type: str,
    mlp_ratio: float,
    proj_bias: bool,
) -> nn.Module:
    if proj_type == "linear":
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=proj_bias,
        )

    if proj_type == "mlp_self":
        hidden_channels = int(in_channels * mlp_ratio)
        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=proj_bias,
            ),
            nn.GELU(),
        ]
        layers.append(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=proj_bias,
            )
        )
        return nn.Sequential(*layers)

    raise ValueError(f"Unknown projection type: {proj_type}")


def tokens_to_2d_feature_map(
    tokens: torch.Tensor,
    bs_flat: int,
    patch_h: int,
    patch_w: int,
) -> torch.Tensor:
    if tokens.dim() != 3:
        raise ValueError(
            "tokens must be [B*S, T, C] before 2D reshape. "
            f"Got shape {tuple(tokens.shape)}"
        )
    if tokens.shape[0] != bs_flat:
        raise ValueError(
            "tokens batch mismatch before 2D reshape: "
            f"expected {bs_flat}, got {tokens.shape[0]}"
        )

    expected_num_patches = patch_h * patch_w
    if tokens.shape[1] != expected_num_patches:
        raise ValueError(
            "tokens patch length mismatch before 2D reshape: "
            f"expected {expected_num_patches}, got {tokens.shape[1]}"
        )

    return tokens.permute(0, 2, 1).reshape(bs_flat, -1, patch_h, patch_w)


class ResidualWrapper(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


def build_post_conv_or_raise(
    post_conv_mode: str,
    post_conv_channels: int,
) -> Optional[nn.Module]:
    if post_conv_mode == "none":
        return None

    if post_conv_mode == "res3x3":
        return ResidualWrapper(
            nn.Conv2d(
                in_channels=post_conv_channels,
                out_channels=post_conv_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            )
        )

    if post_conv_mode == "res5x5":
        return ResidualWrapper(
            nn.Conv2d(
                in_channels=post_conv_channels,
                out_channels=post_conv_channels,
                kernel_size=5,
                padding=2,
                bias=False,
            )
        )

    if post_conv_mode == "res2_3x3_gelu":
        return ResidualWrapper(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=post_conv_channels,
                    out_channels=post_conv_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GELU(),
                nn.Conv2d(
                    in_channels=post_conv_channels,
                    out_channels=post_conv_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
            )
        )

    raise ValueError(
        f"Unknown post_conv_mode: {post_conv_mode}. "
        "Supported: ['none', 'res3x3', 'res5x5', 'res2_3x3_gelu']"
    )

