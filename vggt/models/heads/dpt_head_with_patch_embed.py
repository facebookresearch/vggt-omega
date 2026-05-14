# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Inspired by https://github.com/DepthAnything/Depth-Anything-V2


import os
from typing import List, Dict, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .head_act import activate_head
from .utils import create_uv_grid, position_grid_to_embed


class DPTHead(nn.Module):
    """
    DPT  Head for dense prediction tasks.

    This implementation follows the architecture described in "Vision Transformers for Dense Prediction"
    (https://arxiv.org/abs/2103.13413). The DPT head processes features from a vision transformer
    backbone and produces dense predictions by fusing multi-scale features.

    Args:
        dim_in (int): Input dimension (channels).
        patch_size (int, optional): Patch size. Default is 14.
        output_dim (int, optional): Number of output channels. Default is 4.
        activation (str, optional): Activation type. Default is "inv_log".
        conf_activation (str, optional): Confidence activation type. Default is "expp1".
        features (int, optional): Feature channels for intermediate representations. Default is 256.
        out_channels (List[int], optional): Output channels for each intermediate layer.
        intermediate_layer_idx (List[int], optional): Indices of layers from aggregated tokens used for DPT.
        pos_embed (bool, optional): Whether to use positional embedding. Default is True.
        feature_only (bool, optional): If True, return features only without the last several layers and activation head. Default is False.
        down_ratio (int, optional): Downscaling factor for the output resolution. Default is 1.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 16,
        output_dim: int = 4,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = False,
        feature_only: bool = False,
        down_ratio: int = 1,
        eps = 1e-5,
        disable_conf: bool = False,
        predict_mask: bool = False,
        mask_activation: str = "sigmoid",
        head_features_1: int = None,
        half_dim_in: bool = False,
        **kwargs,
    ) -> None:
        super(DPTHead, self).__init__()
        if len(kwargs) > 0:
            print(f"DPTHead ignored kwargs: {kwargs.keys()}")

        self.half_dim_in = half_dim_in
        if half_dim_in:
            dim_in = dim_in // 2

        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx
        self.disable_conf = disable_conf
        self.predict_mask = predict_mask
        self.mask_activation = mask_activation

        self.norm = nn.LayerNorm(dim_in, eps=eps)

        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )

        self.scratch = _make_scratch(out_channels, features, expand=False)

        # Attach additional modules to scratch.
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = head_features_1 if head_features_1 is not None else features
        head_features_2 = 32

        if feature_only:
            self.scratch.output_conv1 = nn.Conv2d(features, head_features_1, kernel_size=3, stride=1, padding=1)
        else:
            self.scratch.output_conv1 = nn.Conv2d(
                features, head_features_1 // 2, kernel_size=3, stride=1, padding=1
            )
            conv2_in_channels = head_features_1 // 2

            _output_dim = output_dim - 1 if self.disable_conf else output_dim
            if self.predict_mask:
                _output_dim += 1

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, _output_dim, kernel_size=1, stride=1, padding=0),
            )

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = -1,
        patch_embed_intermediate: Optional[List[torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """
        Forward pass through the DPT head, supports processing by chunking frames.
        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
            patch_start_idx (int): Starting index for patch tokens in the token sequence.
                Used to separate patch tokens from other tokens (e.g., camera or register tokens).
            frames_chunk_size (int, optional): Number of frames to process in each chunk.
                If None or larger than S, all frames are processed at once. Default: 8.
            patch_embed_intermediate (List[Tensor], optional): Optional patch-embed branch
                tokens returned by the aggregator. When provided, it is fused as one
                additional DPT branch before aggregated-token branches.

        Returns:
            Tensor or Tuple[Tensor, Tensor, Optional[Tensor]]:
                - If feature_only=True: Feature maps with shape [B, S, C, H, W]
                - Otherwise: Tuple of (predictions, confidence, mask)
                    where mask has shape [B, S, 1, H, W] if predict_mask=True, else None
        """
        B, S, _, H, W = images.shape

        return self._forward_impl(
            aggregated_tokens_list,
            images,
            patch_start_idx,
            patch_embed_intermediate=patch_embed_intermediate,
        )


    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
        patch_embed_intermediate: Optional[List[torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """
        Implementation of the forward pass through the DPT head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.
            patch_embed_intermediate (List[Tensor], optional): Optional patch-embed
                tokens to be used as an extra fusion branch.

        Returns:
            Tensor or Tuple[Tensor, Tensor, Optional[Tensor]]: Feature maps or (predictions, confidence, mask).
        """
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        num_patches = patch_h * patch_w

        token_branches: List[torch.Tensor] = []
        total_seq_len = aggregated_tokens_list[self.intermediate_layer_idx[0]].shape[1]

        if patch_embed_intermediate is not None:
            if not isinstance(patch_embed_intermediate, (list, tuple)) or len(patch_embed_intermediate) != 1:
                raise ValueError(
                    "patch_embed_intermediate must be a list/tuple with exactly one tensor "
                    f"when provided, got {type(patch_embed_intermediate)} with len "
                    f"{len(patch_embed_intermediate) if isinstance(patch_embed_intermediate, (list, tuple)) else 'N/A'}."
                )

            pe_tokens = patch_embed_intermediate[0]
            if pe_tokens.dim() == 3:
                if pe_tokens.shape[0] % B != 0:
                    raise ValueError(
                        "patch_embed_intermediate[0] batch mismatch: expected first dim "
                        f"to be divisible by batch size {B}, got {pe_tokens.shape[0]}."
                    )
                pe_seq_len = pe_tokens.shape[0] // B
                if pe_seq_len != total_seq_len:
                    raise ValueError(
                        "patch_embed_intermediate[0] sequence mismatch: expected "
                        f"{total_seq_len}, got {pe_seq_len}."
                    )
                pe_tokens = pe_tokens.view(B, pe_seq_len, pe_tokens.shape[1], pe_tokens.shape[2])
            elif pe_tokens.dim() == 4:
                if pe_tokens.shape[0] != B or pe_tokens.shape[1] != total_seq_len:
                    raise ValueError(
                        "patch_embed_intermediate[0] must match [B, S, T, C] with "
                        f"B={B}, S={total_seq_len}, got {tuple(pe_tokens.shape)}."
                    )
            else:
                raise ValueError(
                    "patch_embed_intermediate[0] must be [B*S, T, C] or [B, S, T, C], "
                    f"got shape {tuple(pe_tokens.shape)}."
                )

            if pe_tokens.shape[2] < num_patches:
                raise ValueError(
                    "patch_embed_intermediate[0] has too few tokens for patch grid: "
                    f"tokens={pe_tokens.shape[2]}, num_patches={num_patches}."
                )
            # Keep tail patch tokens to stay robust to variable prefix-token counts.
            pe_tokens = pe_tokens[:, :, -num_patches:, :]
            token_branches.append(pe_tokens)

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx]
            if self.half_dim_in:
                x = x[..., : x.shape[-1] // 2]
            if patch_start_idx < 0 or patch_start_idx >= x.shape[2]:
                raise ValueError(
                    f"Invalid patch_start_idx={patch_start_idx} for token length {x.shape[2]} at layer {layer_idx}."
                )
            x = x[:, :, patch_start_idx:]
            if x.shape[2] != num_patches:
                raise ValueError(
                    f"Layer {layer_idx} patch token count mismatch after patch_start_idx: "
                    f"got {x.shape[2]}, expected num_patches({num_patches})."
                )
            token_branches.append(x)

        if len(token_branches) != len(self.projects):
            raise ValueError(
                "Number of fusion branches must match DPT projection branches. "
                f"Got branches={len(token_branches)} and projects={len(self.projects)}. "
                "If patch_embed_intermediate is used, set intermediate_layer_idx to 3 layers."
            )

        out = []
        for dpt_idx, x in enumerate(token_branches):
            if x.dtype != torch.float32:
                x = x.float()

            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            if x.shape[2] != num_patches:
                raise ValueError(
                    f"Branch {dpt_idx} token count mismatch: expected {num_patches}, got {x.shape[2]}."
                )
            x = x.reshape(B * S, -1, x.shape[-1])

            x = self.norm(x)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)

            out.append(x)

        # Fuse features from multiple layers.
        out = self.scratch_forward(out)
        # Interpolate fused output to match target image resolution.
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        if self.feature_only:
            return out.view(B, S, *out.shape[1:])

        feat = self.scratch.output_conv2(out)
        if self.predict_mask:
            feat_mask = feat[:, -1:]
            feat = feat[:, :-1]
        else:
            feat_mask = None

        if self.disable_conf:
            feat = torch.cat([feat, torch.zeros_like(feat[:, :1])], dim=1)

        preds, conf = activate_head(feat, activation=self.activation, conf_activation=self.conf_activation)

        if self.disable_conf:
            conf = torch.ones_like(conf)

        if self.predict_mask:
            if self.mask_activation == "sigmoid":
                mask = torch.sigmoid(feat_mask)
            elif self.mask_activation == "none":
                mask = feat_mask
            else:
                raise ValueError(f"Unknown mask activation: {self.mask_activation}")
        else:
            mask = None

        preds = preds.view(B, S, *preds.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])
        if mask is not None:
            mask = mask.view(B, S, *mask.shape[1:])
        return preds, conf, mask


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

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out


class CheckpointedDPTHead(nn.Module):
    """
    Wrapper around DPTHead that applies gradient checkpointing for memory efficiency.

    Args:
        use_checkpoint (bool): Whether to use gradient checkpointing. Default is True.
        **kwargs: Arguments passed to DPTHead.
    """

    def __init__(self, use_checkpoint: bool = True, **kwargs) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.head = DPTHead(**kwargs)

    def forward(self, *args, **kwargs):
        if self.use_checkpoint and self.training:
            return checkpoint(self.head, *args, use_reentrant=False, **kwargs)
        return self.head(*args, **kwargs)


################################################################################
# Modules
################################################################################


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
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
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)
