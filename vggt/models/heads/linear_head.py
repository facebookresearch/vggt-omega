# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional, Tuple, Union

import contextlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .head_act import activate_head
from vggt.models.layers.ffn_layers import Mlp
from vggt.models.layers import SelfAttentionBlock, RopePositionEmbedding

logger = logging.getLogger(__name__)

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

class LinearHead(nn.Module):
    """
    Linear Head for dense prediction tasks.

    This implementation provides a simple linear transformation approach for dense predictions.
    Features from vision transformer backbone are processed through layer normalization and 
    linear projection, then reshaped to produce dense predictions using pixel shuffle.

    Args:
        dim_in (int): Input dimension (channels).
        patch_size (int, optional): Patch size. Default is 16.
        output_dim (int, optional): Number of output channels. Default is 4.
        activation (str, optional): Activation type. Default is "norm_exp".
        conf_activation (str, optional): Confidence activation type. Default is "norm_exp".
        intermediate_layer_idx (List[int], optional): Indices of layers from aggregated tokens used for fusion.
            If None, uses only the last layer. Default is None.
        disable_last_layer_amp (bool, optional): Disable AMP in the projection MLP/linear head. Default is False.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 16,
        output_dim: int = 4,
        activation: str = "norm_exp",
        conf_activation: str = "norm_exp",
        intermediate_layer_idx: List[int] = None,
        mlp_ratio: float = 2.0,
        proj_type = "linear",
        proj_bias: bool = True,
        proj_zero_init: bool = False,
        eps = 1e-5,
        fuse_type: str = "mlp",
        disable_conf: bool = False,
        separate_conf_head: bool = False,
        predict_mask: bool = False,
        mask_activation: str = "sigmoid",
        # Extra attention blocks
        extra_attention_depth: int = 0,
        extra_attention_dim: int = 1024,  # Project to this dimension before attention (like Pi3's TransformerDecoder)
        extra_attention_num_heads: int = 16,
        extra_attention_mlp_ratio: float = 4.0,
        extra_attention_qkv_bias: bool = True,
        extra_attention_proj_bias: bool = True,
        extra_attention_ffn_bias: bool = True,
        extra_attention_use_qk_norm: bool = True,
        extra_attention_init_values: float = 1e-5,
        extra_attention_mask_k_bias: bool = True,
        extra_attention_post_norm: bool = True,  # Whether to apply LayerNorm after extra attention blocks
        # RoPE for extra attention blocks
        extra_attention_rope_freq: Optional[int] = 100,
        extra_attention_rope_dtype: str = "fp32",
        extra_attention_rope_rescale_coords: Optional[float] = None,
        extra_attention_rope_normalize_coords: str = "separate",
        extra_attention_rope_coord_norm_denominator: Optional[int] = None,
        disable_last_layer_amp: bool = True,
        use_checkpoint: bool = True,
        use_flash_attn: bool = False,
        **kwargs,
    ) -> None:
        super(LinearHead, self).__init__()
        
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.activation = activation
        self.conf_activation = conf_activation
        self.intermediate_layer_idx = intermediate_layer_idx
        self.fuse_type = fuse_type
        self.disable_conf = disable_conf
        if separate_conf_head:
            assert disable_conf, "separate_conf_head is only valid when disable_conf=True"
        self.separate_conf_head = separate_conf_head
        self.predict_mask = predict_mask
        self.mask_activation = mask_activation
        self.disable_last_layer_amp = disable_last_layer_amp
        assert self.disable_last_layer_amp, (
            "LinearHead output head must run in fp32. "
            "Set disable_last_layer_amp=True."
        )
        self.use_checkpoint = use_checkpoint
        self.use_reentrant = False
        self.extra_attention_use_qk_norm = extra_attention_use_qk_norm

        if extra_attention_depth > 0:
            self.extra_attention_dim = extra_attention_dim
            # Project from dim_in to extra_attention_dim before attention (like Pi3's TransformerDecoder)
            self.extra_attention_in_proj = nn.Linear(dim_in, extra_attention_dim)
            self.extra_attention_blocks = nn.ModuleList([
                SelfAttentionBlock(
                    dim=extra_attention_dim,
                    num_heads=extra_attention_num_heads,
                    ffn_ratio=extra_attention_mlp_ratio,
                    qkv_bias=extra_attention_qkv_bias,
                    proj_bias=extra_attention_proj_bias,
                    ffn_bias=extra_attention_ffn_bias,
                    init_values=extra_attention_init_values,
                    use_qk_norm=extra_attention_use_qk_norm,
                    mask_k_bias=extra_attention_mask_k_bias,
                    use_flash_attn=use_flash_attn,
                ) for _ in range(extra_attention_depth)
            ])
            # Initialize RoPE for extra attention if enabled
            if extra_attention_rope_freq is not None and extra_attention_rope_freq > 0:
                self.extra_attention_rope_dtype = dtype_dict[extra_attention_rope_dtype]
                self.extra_attention_rope_embed = RopePositionEmbedding(
                    embed_dim=extra_attention_dim,
                    num_heads=extra_attention_num_heads,
                    base=extra_attention_rope_freq,
                    rescale_coords=extra_attention_rope_rescale_coords,
                    normalize_coords=extra_attention_rope_normalize_coords,
                    coord_norm_denominator=extra_attention_rope_coord_norm_denominator,
                    dtype=self.extra_attention_rope_dtype,
                )
                if extra_attention_rope_coord_norm_denominator is not None:
                    logger.info(
                        "LinearHead: using extra_attention RoPE coord_norm_denominator=%s",
                        extra_attention_rope_coord_norm_denominator,
                    )
            else:
                self.extra_attention_rope_embed = None
        else:
            self.extra_attention_dim = None
            self.extra_attention_in_proj = None
            self.extra_attention_blocks = None
            self.extra_attention_rope_embed = None

        # Determine the feature dimension after optional extra attention
        # If extra attention is used, features are projected to extra_attention_dim
        feature_dim = extra_attention_dim if extra_attention_depth > 0 else dim_in

        # LayerNorm after extra attention blocks (Pi3 doesn't have this, so can be disabled)
        if extra_attention_post_norm:
            self.norm = nn.LayerNorm(feature_dim, eps=eps)
        else:
            self.norm = nn.Identity()
            
        # Determine input dimension for projection based on layer fusion
        if intermediate_layer_idx is not None:
            if self.fuse_type == "mlp":
                proj_input_dim = feature_dim * len(intermediate_layer_idx)
            elif self.fuse_type == "add":
                proj_input_dim = feature_dim
            else:
                raise ValueError(f"Unknown fuse type: {self.fuse_type}")
        else:
            proj_input_dim = feature_dim
            
        _output_dim = output_dim - 1 if self.disable_conf else output_dim
            
        if proj_type == "linear":
            self.proj = nn.Linear(proj_input_dim, _output_dim * self.patch_size ** 2, bias=proj_bias)
        elif proj_type == "mlp":
            self.proj = nn.Sequential(
                Mlp(proj_input_dim, int(proj_input_dim * mlp_ratio)),
                nn.LayerNorm(proj_input_dim, eps=eps),
                nn.Linear(proj_input_dim, _output_dim * self.patch_size ** 2),
            )
        elif proj_type == "mlp_self":
            self.proj = Mlp(proj_input_dim, int(proj_input_dim * mlp_ratio), _output_dim * self.patch_size ** 2)
        else:
            raise ValueError(f"Unknown projection type: {proj_type}")
        
        # Build separate confidence head when disable_conf=True and separate_conf_head=True
        if self.separate_conf_head:
            conf_output_dim = 1
            if proj_type == "linear":
                self.proj_conf = nn.Linear(proj_input_dim, conf_output_dim * self.patch_size ** 2, bias=proj_bias)
            elif proj_type == "mlp":
                self.proj_conf = nn.Sequential(
                    Mlp(proj_input_dim, int(proj_input_dim * mlp_ratio)),
                    nn.LayerNorm(proj_input_dim, eps=eps),
                    nn.Linear(proj_input_dim, conf_output_dim * self.patch_size ** 2),
                )
            elif proj_type == "mlp_self":
                self.proj_conf = Mlp(proj_input_dim, int(proj_input_dim * mlp_ratio), conf_output_dim * self.patch_size ** 2)
        else:
            self.proj_conf = None

        # Build mask prediction head when predict_mask=True
        if self.predict_mask:
            mask_output_dim = 1
            if proj_type == "linear":
                self.proj_mask = nn.Linear(proj_input_dim, mask_output_dim * self.patch_size ** 2, bias=proj_bias)
            elif proj_type == "mlp":
                self.proj_mask = nn.Sequential(
                    Mlp(proj_input_dim, int(proj_input_dim * mlp_ratio)),
                    nn.LayerNorm(proj_input_dim, eps=eps),
                    nn.Linear(proj_input_dim, mask_output_dim * self.patch_size ** 2),
                )
            elif proj_type == "mlp_self":
                self.proj_mask = Mlp(proj_input_dim, int(proj_input_dim * mlp_ratio), mask_output_dim * self.patch_size ** 2)
        else:
            self.proj_mask = None

        # Zero initialize the last layer weights and biases if enabled
        if proj_zero_init:
            self._zero_init_last_layer(self.proj, proj_type)
            if self.proj_conf is not None:
                self._zero_init_last_layer(self.proj_conf, proj_type)
            if self.proj_mask is not None:
                self._zero_init_last_layer(self.proj_mask, proj_type)

    def _zero_init_last_layer(self, proj: nn.Module, proj_type: str) -> None:
        """Zero initialize the weight and bias of the last layer in a projection module."""
        if proj_type == "linear":
            last_layer = proj
        elif proj_type == "mlp":
            last_layer = proj[-1]  # Last layer in Sequential
        elif proj_type == "mlp_self":
            last_layer = proj.fc2  # Mlp has fc2 as output layer
        else:
            return
        
        nn.init.zeros_(last_layer.weight)
        if last_layer.bias is not None:
            nn.init.zeros_(last_layer.bias)

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = -1,
        patch_embed_intermediate: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the Linear head.
        
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
        Implementation of the forward pass through the Linear head.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tuple[Tensor, Tensor, Optional[Tensor]]: Tuple of (predictions, confidence, mask).
        """
        assert frames_start_idx is None and frames_end_idx is None, "Linear head does not support chunking"
        
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape

        # Use only the last layer (keep all tokens for extra attention)
        tokens = aggregated_tokens_list[-1]
        
        # Select frames if processing a chunk
        if frames_start_idx is not None and frames_end_idx is not None:
            tokens = tokens[:, frames_start_idx:frames_end_idx]
        
        _, _, T, C = tokens.shape
        tokens = tokens.view(B * S, T, C)
        
        if self.extra_attention_blocks is not None:
            # Project to extra_attention_dim before attention (like Pi3's TransformerDecoder)
            tokens = self.extra_attention_in_proj(tokens)
            
            # Compute RoPE if enabled
            rope_sincos = None
            if self.extra_attention_rope_embed is not None:
                with torch.no_grad():
                    rope_sin, rope_cos = self.extra_attention_rope_embed(
                        H=H // self.patch_size, W=W // self.patch_size
                    )
                    rope_sincos = (
                        rope_sin.to(device=tokens.device, dtype=self.extra_attention_rope_dtype),
                        rope_cos.to(device=tokens.device, dtype=self.extra_attention_rope_dtype),
                    )
            for blk in self.extra_attention_blocks:
                if self.training and self.use_checkpoint:
                    tokens = checkpoint(blk, tokens, rope_sincos, use_reentrant=self.use_reentrant)
                else:
                    tokens = blk(tokens, rope_sincos)

        # Extract patch tokens only (after extra attention if used)
        tokens = tokens[:, patch_start_idx:]

        # tokens = self.norm(tokens)

        last_layer_amp_context = (
            torch.cuda.amp.autocast(enabled=False)
            if self.disable_last_layer_amp
            else contextlib.nullcontext()
        )
        
        with last_layer_amp_context:
            if self.disable_last_layer_amp:
                tokens = tokens.float()

            tokens = self.norm(tokens)
            feat = self.proj(tokens)
            
            # Use separate confidence head if enabled
            if self.separate_conf_head:
                feat_conf = self.proj_conf(tokens)

            # Use mask prediction head if enabled
            if self.predict_mask:
                feat_mask = self.proj_mask(tokens)
            
            # Reshape and apply pixel shuffle
            feat = feat.transpose(-1, -2).contiguous()
            feat = feat.view(B * S, -1, H // self.patch_size, W // self.patch_size)
            feat = F.pixel_shuffle(feat, self.patch_size)  # [B*S, output_dim, H, W]
            
            # Reshape confidence features if using separate head
            if self.separate_conf_head:
                feat_conf = feat_conf.transpose(-1, -2).contiguous()
                feat_conf = feat_conf.view(B * S, -1, H // self.patch_size, W // self.patch_size)
                feat_conf = F.pixel_shuffle(feat_conf, self.patch_size)  # [B*S, 1, H, W]

            # Reshape mask features if using mask prediction head
            if self.predict_mask:
                feat_mask = feat_mask.transpose(-1, -2).contiguous()
                feat_mask = feat_mask.view(B * S, -1, H // self.patch_size, W // self.patch_size)
                feat_mask = F.pixel_shuffle(feat_mask, self.patch_size)  # [B*S, 1, H, W]

            # Handle disable_conf: append dummy channel for activation, then overwrite conf
            if self.disable_conf and not self.separate_conf_head:
                # feat is [B*S, output_dim-1, H, W]
                # append dummy channel to match output_dim for activate_head
                feat = torch.cat([feat, torch.zeros_like(feat[:, :1])], dim=1)

            if self.activation == "none":
                preds = feat.permute(0, 2, 3, 1) # [B*S, H, W, C]
                conf = torch.ones_like(preds[..., :1]) # Dummy confidence
            else:
                if self.separate_conf_head:
                    # Concatenate pred features with conf features for activate_head
                    feat_combined = torch.cat([feat, feat_conf], dim=1)
                    preds, conf = activate_head(feat_combined, activation=self.activation, conf_activation=self.conf_activation)
                else:
                    preds, conf = activate_head(feat, activation=self.activation, conf_activation=self.conf_activation)
                
            if self.disable_conf and not self.separate_conf_head:
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

            # Reshape back to [B, S, ...]
            preds = preds.view(B, S, *preds.shape[1:])
            conf = conf.view(B, S, *conf.shape[1:])
            if mask is not None:
                mask = mask.view(B, S, *mask.shape[1:])

            assert preds.dtype == torch.float32 and conf.dtype == torch.float32, (
                f"LinearHead outputs must be fp32, got preds={preds.dtype}, conf={conf.dtype}"
            )

            return preds, conf, mask


