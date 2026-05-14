# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn


SUPPORTED_MULTISCALE_RESIZE_TYPES = (
    "deconv",
    "bilinear",
    "bilinear_dw",
    "bilinear_normal_conv",
)

SUPPORTED_POS_EMBED_TYPES = (
    "none",
    "rope2d",
    "rope4d",
)
# Backward-compatible alias.
SUPPORTED_MULTISCALE_POS_EMBED_TYPES = SUPPORTED_POS_EMBED_TYPES


def build_multiscale_resize_layer(
    out_channels: int,
    resize_type: str,
    upsample_factor: int,
) -> nn.Module:
    if resize_type == "deconv":
        return nn.ConvTranspose2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=upsample_factor,
            stride=upsample_factor,
            padding=0,
        )
    if resize_type == "bilinear":
        return nn.Upsample(
            scale_factor=upsample_factor,
            mode="bilinear",
            align_corners=False,
        )
    if resize_type == "bilinear_dw":
        return nn.Sequential(
            nn.Upsample(
                scale_factor=upsample_factor,
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
        )
    if resize_type == "bilinear_normal_conv":
        return nn.Sequential(
            nn.Upsample(
                scale_factor=upsample_factor,
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )

    raise ValueError(
        f"Unknown multiscale_resize_type: {resize_type}. "
        f"Supported: {list(SUPPORTED_MULTISCALE_RESIZE_TYPES)}"
    )


def build_rope_like_pos_embed(
    *,
    height: int,
    width: int,
    channels: int,
    pos_embed_type: str,
    base: float = 100.0,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a RoPE-like sin/cos positional embedding map with coords in [-1, 1]."""
    if pos_embed_type not in SUPPORTED_POS_EMBED_TYPES:
        raise ValueError(
            f"Unknown pos_embed_type: {pos_embed_type}. "
            f"Supported: {list(SUPPORTED_POS_EMBED_TYPES)}"
        )
    if pos_embed_type == "none":
        return torch.zeros((1, channels, height, width), dtype=dtype, device=device)
    if base <= 0:
        raise ValueError(f"base must be > 0, got {base}")

    coords_h = (torch.arange(0.5, height, device=device, dtype=dtype) / height) * 2.0 - 1.0
    coords_w = (torch.arange(0.5, width, device=device, dtype=dtype) / width) * 2.0 - 1.0
    grid_h, grid_w = torch.meshgrid(coords_h, coords_w, indexing="ij")

    if pos_embed_type == "rope2d":
        coords = torch.stack([grid_h, grid_w], dim=-1)  # [H, W, 2]
    else:
        coords = torch.stack(
            [
                grid_h,
                grid_w,
                0.5 * (grid_h + grid_w),
                0.5 * (grid_h - grid_w),
            ],
            dim=-1,
        )  # [H, W, 4]

    coord_dim = coords.shape[-1]
    freq_dim = max(1, math.ceil(channels / (2 * coord_dim)))
    periods = base ** (torch.arange(freq_dim, device=device, dtype=dtype) / freq_dim)
    angles = (2.0 * math.pi * coords[..., None]) / periods  # [H, W, coord_dim, freq_dim]
    embed = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).reshape(height, width, -1)
    embed = embed[..., :channels]
    return embed.permute(2, 0, 1).unsqueeze(0).contiguous()
