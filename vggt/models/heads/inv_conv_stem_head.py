# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from .head_act import activate_head


class InvConvStemHead(nn.Module):
    """
    Inverse Convolution Stem Head for dense prediction tasks.

    This head uses an inverse convolution stem architecture that mirrors a typical
    ConvStem with 4× stride-2 convs + 1× conv-1×1. It processes features from a
    vision transformer backbone and produces dense predictions through transposed
    convolutions.

    Args:
        dim_in (int): Input dimension (channels).
        patch_size (int, optional): Patch size. Default is 16.
        output_dim (int, optional): Number of output channels. Default is 4.
        activation (str, optional): Activation type. Default is "inv_log".
        conf_activation (str, optional): Confidence activation type. Default is "expp1".
        intermediate_layer_idx (int, optional): Index of layer from aggregated tokens used for prediction.
            Default is -1 (last layer).
        eps (float, optional): Epsilon for LayerNorm. Default is 1e-5.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 16,
        output_dim: int = 4,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        intermediate_layer_idx: int = -1,
        eps: float = 1e-5,
        disable_conf: bool = False,
        predict_mask: bool = False,
        mask_activation: str = "sigmoid",
        **kwargs,
    ) -> None:
        super(InvConvStemHead, self).__init__()
        if len(kwargs) > 0:
            print(f"InvConvStemHead ignored kwargs: {kwargs.keys()}")

        assert patch_size == 16, "InvConvStemHead requires patch size 16 for symmetry"
        assert dim_in % 8 == 0, "dim_in must be divisible by 8"

        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.intermediate_layer_idx = intermediate_layer_idx
        self.disable_conf = disable_conf
        self.predict_mask = predict_mask
        self.mask_activation = mask_activation

        self.norm = nn.LayerNorm(dim_in, eps=eps)

        _output_dim = output_dim - 1 if self.disable_conf else output_dim
        if self.predict_mask:
            _output_dim += 1

        # Build the inverse convolution stem
        self.inv_stem = InvConvStem(
            embed_dim=dim_in,
            out_chans=_output_dim,
        )

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the InvConvStem head, supports processing by chunking frames.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
            patch_start_idx (int): Starting index for patch tokens in the token sequence.
                Used to separate patch tokens from other tokens (e.g., camera or register tokens).
            frames_chunk_size (int, optional): Number of frames to process in each chunk.
                If None or larger than S, all frames are processed at once. Default: -1.

        Returns:
            Tuple[Tensor, Tensor, Optional[Tensor]]: Tuple of (predictions, confidence, mask).
                predictions and confidence have shape [B, S, output_dim, H, W].
                mask has shape [B, S, 1, H, W] if predict_mask=True, else None.
        """
        B, S, _, H, W = images.shape

        return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Implementation of the forward pass through the InvConvStem head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tuple[Tensor, Tensor, Optional[Tensor]]: Tuple of (predictions, confidence, mask).
        """
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # Get tokens from the specified layer
        x = aggregated_tokens_list[self.intermediate_layer_idx][:, :, patch_start_idx:]

        # Select frames if processing a chunk
        if frames_start_idx is not None and frames_end_idx is not None:
            x = x[:, frames_start_idx:frames_end_idx]

        # Reshape: [B, S, N, C] -> [B*S, N, C]
        x = x.reshape(B * S, -1, x.shape[-1])

        # Apply layer norm
        x = self.norm(x)

        # Reshape to spatial: [B*S, N, C] -> [B*S, C, patch_h, patch_w]
        C = x.shape[-1]
        N = x.shape[1]
        assert N == patch_h * patch_w, f"Token count {N} != spatial resolution {patch_h}x{patch_w} ({patch_h*patch_w}). Check patch_start_idx/image size."
        x = x.permute(0, 2, 1).reshape((x.shape[0], C, patch_h, patch_w))

        # Apply inverse convolution stem
        out = self.inv_stem(x)

        if self.predict_mask:
            feat_mask = out[:, -1:]
            out = out[:, :-1]
        else:
            feat_mask = None

        if self.disable_conf:
            out = torch.cat([out, torch.zeros_like(out[:, :1])], dim=1)

        # Apply activation head to get predictions and confidence
        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)

        if self.disable_conf:
            conf = torch.ones_like(conf)

        # Apply mask activation if using mask prediction head
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


################################################################################
# Modules
################################################################################


class InvConvStem(nn.Module):
    """
    Inverse of ConvStem with GroupNorm.

    This module performs the inverse operation of a typical ConvStem used in
    vision transformers. It upsamples feature maps from (B, embed_dim, H/16, W/16)
    to (B, out_chans, H, W) using transposed convolutions.

    Assumes the forward stem used 4× stride-2 convs + 1× conv-1×1.

    Args:
        embed_dim (int, optional): Input embedding dimension. Default is 1024.
        out_chans (int, optional): Number of output channels. Default is 4.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        out_chans: int = 4,
    ) -> None:
        super(InvConvStem, self).__init__()

        assert embed_dim % 8 == 0, "embed_dim must be divisible by 8"

        self.embed_dim = embed_dim
        self.out_chans = out_chans

        layers = []
        in_dim = embed_dim

        # 4 stages of transposed convolution, each doubles spatial resolution and halves channels
        for _ in range(4):
            out_dim = in_dim // 2
            layers += [
                nn.ConvTranspose2d(
                    in_dim, out_dim,
                    kernel_size=3, stride=2, padding=1, output_padding=1, bias=True
                ),
                nn.GroupNorm(8, out_dim),
                nn.ReLU(inplace=True),
            ]
            in_dim = out_dim  # halve channels each stage

        # Final 1x1 convolution for output channels
        layers += [nn.Conv2d(in_dim, out_chans, kernel_size=1)]

        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the inverse convolution stem.

        Args:
            x (Tensor): Input tensor with shape [B, embed_dim, H/16, W/16].

        Returns:
            Tensor: Output tensor with shape [B, out_chans, H, W].
        """
        return self.proj(x)
