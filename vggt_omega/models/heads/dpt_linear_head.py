# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Inspired by https://github.com/DepthAnything/Depth-Anything-V2


import contextlib
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from .head_act import activate_head
from .utils import create_uv_grid, position_grid_to_embed


_CONF_PROJ_INIT_CONF_VALUE = 1.05
_CONF_PROJ_INIT_RAW_BIAS = math.log(_CONF_PROJ_INIT_CONF_VALUE - 1.0)


class DPTLinearHead(nn.Module):
    """
    DPT Head with a linear output decoder.

    This implementation intentionally stays close to DPTHead. The multi-scale token projection,
    resize path, and scratch fusion are unchanged. The only architecture change is that the fused
    feature stays at layer_1 scale, and a lightweight linear head + pixel shuffle predicts the
    final full-resolution output.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        patch_size: int = 16,
        output_dim: int = 2,
        activation: str = "exp",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        eps = 1e-5,
        disable_last_layer_amp: bool = True,
        proj_type: str = "linear",
        mlp_ratio: float = 0.5,
        proj_bias: bool = True,
        fusion_block_relu_inplace: bool = False,
        use_interpolate_conv2d_upsample: bool = False,
    ) -> None:
        super(DPTLinearHead, self).__init__()

        if patch_size % 4 != 0:
            raise ValueError(
                "DPTLinearHead expects patch_size divisible by 4 because the fused feature is decoded "
                f"from 1/4 scale. Got patch_size={patch_size}."
            )

        self.patch_size = patch_size
        self.output_dim = output_dim
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.intermediate_layer_idx = intermediate_layer_idx
        self.disable_last_layer_amp = disable_last_layer_amp
        self.final_shuffle_factor = patch_size // 4
        self.use_interpolate_conv2d_upsample = use_interpolate_conv2d_upsample

        self.norm = nn.LayerNorm(dim_in, eps=eps)

        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
                _make_dpt_resize_layer(
                    channels=out_channels[0],
                    resize_scale=4.0,
                    use_interpolate_conv2d_upsample=self.use_interpolate_conv2d_upsample,
                ),
                _make_dpt_resize_layer(
                    channels=out_channels[1],
                    resize_scale=2.0,
                    use_interpolate_conv2d_upsample=self.use_interpolate_conv2d_upsample,
                ),
                _make_dpt_resize_layer(
                    channels=out_channels[2],
                    resize_scale=1.0,
                    use_interpolate_conv2d_upsample=self.use_interpolate_conv2d_upsample,
                ),
                _make_dpt_resize_layer(
                    channels=out_channels[3],
                    resize_scale=0.5,
                    use_interpolate_conv2d_upsample=self.use_interpolate_conv2d_upsample,
                ),
            ]
        )

        self.scratch = _make_scratch(out_channels, features, expand=False)

        # Attach additional modules to scratch.
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features, relu_inplace=fusion_block_relu_inplace)
        self.scratch.refinenet2 = _make_fusion_block(features, relu_inplace=fusion_block_relu_inplace)
        self.scratch.refinenet3 = _make_fusion_block(features, relu_inplace=fusion_block_relu_inplace)
        self.scratch.refinenet4 = _make_fusion_block(
            features,
            has_residual=False,
            relu_inplace=fusion_block_relu_inplace,
        )

        proj_in_channels = features
        pred_channels = output_dim - 1

        self.proj = _make_prediction_head(
            proj_in_channels,
            pred_channels * self.final_shuffle_factor ** 2,
            proj_type=proj_type,
            mlp_ratio=mlp_ratio,
            proj_bias=proj_bias,
        )

        self.proj_conf = _make_prediction_head(
            proj_in_channels,
            self.final_shuffle_factor ** 2,
            proj_type=proj_type,
            mlp_ratio=mlp_ratio,
            proj_bias=proj_bias,
        )
        _init_small_conf_prediction_head(self.proj_conf)

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the DPT linear head, supports processing by chunking frames.
        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
            patch_start_idx (int): Starting index for patch tokens in the token sequence.
                Used to separate patch tokens from other tokens (e.g., camera or register tokens).
        Returns:
            Tuple of predicted depth and confidence tensors.
        """
        B, S, _, H, W = images.shape
        del B, S, H, W

        return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)


    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implementation of the forward pass through the DPT linear head.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.

        Returns:
            Tuple of predicted depth and confidence tensors.
        """
        B, S, _, H, W = images.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        last_layer_amp_context = (
            torch.autocast(device_type="cuda", enabled=False)
            if self.disable_last_layer_amp
            else contextlib.nullcontext()
        )

        with last_layer_amp_context:
            out = []
            dpt_idx = 0

            for layer_idx in self.intermediate_layer_idx:
                x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
                if self.disable_last_layer_amp and x.dtype != torch.float32:
                    x = x.float()

                x = x.reshape(B * S, -1, x.shape[-1])

                x = self.norm(x)

                x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

                x = self.projects[dpt_idx](x)
                if self.pos_embed:
                    x = self._apply_pos_embed(x, W, H)
                x = self.resize_layers[dpt_idx](x)

                out.append(x)
                dpt_idx += 1

            # Fuse features from multiple layers.
            out = self.scratch_forward(out)

            if self.pos_embed:
                out = self._apply_pos_embed(out, W, H)

            feat = self.proj(out)
            feat = F.pixel_shuffle(feat, self.final_shuffle_factor)

            feat_conf = self.proj_conf(out)
            feat_conf = F.pixel_shuffle(feat_conf, self.final_shuffle_factor)
            feat = torch.cat([feat, feat_conf], dim=1)

            preds, conf = activate_head(feat, activation=self.activation, conf_activation=self.conf_activation)

            preds = preds.view(B, S, *preds.shape[1:])
            conf = conf.view(B, S, *conf.shape[1:])

            if self.disable_last_layer_amp:
                assert preds.dtype == torch.float32 and conf.dtype == torch.float32, (
                    f"DPTLinearHead outputs must be fp32, got preds={preds.dtype}, conf={conf.dtype}"
                )

            return preds, conf


    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        # Keep output at layer_1 scale (1/4 for patch_size=16).
        out = self.scratch.refinenet1(out, layer_1_rn, size=layer_1_rn.shape[2:])
        del layer_1_rn, layer_1

        return out


################################################################################
# Modules
################################################################################


def _make_dpt_resize_layer(
    channels: int,
    resize_scale: float,
    use_interpolate_conv2d_upsample: bool = False,
    align_corners: bool = True,
) -> nn.Module:
    if resize_scale not in (0.5, 1.0, 2.0, 4.0):
        raise ValueError(
            "Unsupported resize_scale for DPT resize layer. "
            f"Expected one of [0.5, 1.0, 2.0, 4.0], got {resize_scale}."
        )

    if resize_scale == 1.0:
        return nn.Identity()

    if resize_scale < 1.0:
        downsample_stride = int(1.0 / resize_scale)
        return nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=downsample_stride,
            padding=1,
        )

    upsample_scale = int(resize_scale)
    if use_interpolate_conv2d_upsample:
        return nn.Sequential(
            nn.Upsample(
                scale_factor=upsample_scale,
                mode="bilinear",
                align_corners=align_corners,
            ),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    return nn.ConvTranspose2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=upsample_scale,
        stride=upsample_scale,
        padding=0,
    )


def _make_prediction_head(
    in_channels: int,
    out_channels: int,
    proj_type: str = "linear",
    mlp_ratio: float = 2.0,
    proj_bias: bool = True,
) -> nn.Module:
    if proj_type == "linear":
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=proj_bias)
    elif proj_type == "mlp_self":
        hidden_channels = int(in_channels * mlp_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=proj_bias),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=proj_bias),
        )
    elif proj_type == "mlp":
        hidden_channels = int(in_channels * mlp_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=proj_bias),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=proj_bias),
        )
    else:
        raise ValueError(f"Unknown projection type: {proj_type}")


def _init_small_conf_prediction_head(proj: nn.Module) -> None:
    if isinstance(proj, nn.Sequential):
        last_layer = proj[-1]
    else:
        last_layer = proj

    if not isinstance(last_layer, nn.Conv2d):
        raise TypeError(f"Unsupported confidence projection layer: {type(last_layer)}")

    nn.init.zeros_(last_layer.weight)
    if last_layer.bias is None:
        raise ValueError("Small confidence init requires proj_bias=True for proj_conf")

    # With expp1 confidence activation this starts from conf ~= 1.05.
    nn.init.constant_(last_layer.bias, _CONF_PROJ_INIT_RAW_BIAS)


def _make_fusion_block(
    features: int,
    size: int = None,
    has_residual: bool = True,
    groups: int = 1,
    relu_inplace: bool = True,
) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=relu_inplace),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn, groups=1):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    if size is None:
        size = (
            int(x.shape[-2] * scale_factor),
            int(x.shape[-1] * scale_factor),
        )

    if tuple(x.shape[-2:]) == tuple(size):
        return x

    int_max = 1610612736
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > int_max:
        chunks = torch.chunk(x, chunks=(input_elements // int_max) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(
                chunk,
                size=size,
                mode=mode,
                align_corners=align_corners,
            )
            for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()

    return nn.functional.interpolate(
        x,
        size=size,
        mode=mode,
        align_corners=align_corners,
    )
