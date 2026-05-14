# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from .head_utils import (
    SUPPORTED_POS_EMBED_TYPES,
    build_multiscale_resize_layer,
    build_rope_like_pos_embed,
)
from ..head_act import activate_head
from vggt.models.layers.ffn_layers import Mlp

class VGGLinearHead(nn.Module):
    """
    Simplified Linear Head for dense prediction tasks.

    This implementation acts as a clean template, retaining only the active 
    configuration from omegav1_linear_t4.yaml:
    - Linear projection (no extra attention, no MLP)
    - Separate confidence head
    - Mask prediction head
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 16,
        output_dim: int = 4,  # e.g., 3 for depth + 1 for conf
        activation: str = "norm_exp",
        conf_activation: str = "softplus_0.5_10",
        predict_mask: bool = True,
        mask_activation: str = "none",
        proj_type: str = "linear",
        mlp_ratio: float = 2.0,
        proj_bias: bool = True,
        proj_zero_init: bool = False,
        intermediate_layer_idx: Optional[List[int]] = None,
        multiscale_out_channels: Optional[List[int]] = None,
        multiscale_upsample_factor: int = 4,
        multiscale_resize_type: str = "deconv",
        multiscale_merge_type: str = "concat",
        multiscale_norm_type: str = "shared",
        multiscale_head_conv_type: str = "none",
        use_patch_embed_intermediate: bool = False,
        patch_embed_merge_type: str = "concat",
        patch_embed_dim: Optional[int] = None,
        patch_embed_out_channels: Optional[int] = None,
        pos_embed_type: str = "none",
        pos_embed_base: float = 100.0,
        pos_embed_scale: float = 0.1,
        post_conv_type: str = "none",
        eps: float = 1e-5,
        disable_last_layer_amp: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        legacy_pos_embed_keys = (
            "multiscale_pos_embed_type",
            "multiscale_pos_embed_base",
            "multiscale_pos_embed_scale",
        )
        legacy_in_kwargs = [key for key in legacy_pos_embed_keys if key in kwargs]
        if legacy_in_kwargs:
            raise ValueError(
                f"Deprecated pos-embed keys are not supported: {legacy_in_kwargs}. "
                "Use ['pos_embed_type', 'pos_embed_base', 'pos_embed_scale'] instead."
            )
        if "patch_embed_scale_init" in kwargs:
            raise ValueError(
                "Deprecated key 'patch_embed_scale_init' is not supported. "
                "Remove it from config; patch-embed scale gating has been removed."
            )
        
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.activation = activation
        self.conf_activation = conf_activation
        self.predict_mask = predict_mask
        self.mask_activation = mask_activation
        self.proj_type = proj_type
        self.disable_last_layer_amp = disable_last_layer_amp
        self.intermediate_layer_idx = intermediate_layer_idx
        self.use_multiscale = intermediate_layer_idx is not None and len(intermediate_layer_idx) > 0
        self.post_conv_type = post_conv_type
        self.multiscale_head_conv_type = multiscale_head_conv_type
        self.use_patch_embed_intermediate = use_patch_embed_intermediate
        self.patch_embed_merge_type = patch_embed_merge_type
        self.patch_embed_out_channels = patch_embed_out_channels

        assert self.disable_last_layer_amp, (
            "LinearHead output head must run in fp32. "
            "Set disable_last_layer_amp=True."
        )
        if pos_embed_type not in SUPPORTED_POS_EMBED_TYPES:
            raise ValueError(
                f"Unknown pos_embed_type: {pos_embed_type}. "
                f"Supported: {list(SUPPORTED_POS_EMBED_TYPES)}"
            )
        if pos_embed_base <= 0:
            raise ValueError(f"pos_embed_base must be > 0, got {pos_embed_base}")
        self.pos_embed_type = pos_embed_type
        self.pos_embed_base = pos_embed_base
        self.pos_embed_scale = pos_embed_scale
        if self.multiscale_head_conv_type not in ("none", "dwconv", "conv"):
            raise ValueError(
                f"Unknown multiscale_head_conv_type: {self.multiscale_head_conv_type}. "
                "Supported: ['none', 'dwconv', 'conv']"
            )
        self._validate_patch_embed_config(patch_embed_dim=patch_embed_dim)
        self.patch_embed_dim = patch_embed_dim
        self._reset_patch_embed_modules()

        # Avoid creating unused LayerNorm parameters in multiscale+per_level mode.
        # This prevents potential DDP "unused parameter" issues when find_unused_parameters=False.
        if self.use_multiscale and multiscale_norm_type == "per_level":
            self.norm = nn.Identity()
        else:
            self.norm = nn.LayerNorm(dim_in, eps=eps)
            
        # Hardcoding the separated heads logic from the config
        _output_dim = output_dim - 1
        post_conv_channels = _output_dim + 1 + (1 if self.predict_mask else 0)
        if self.post_conv_type == "none":
            self.post_conv = None
        elif self.post_conv_type == "dw_conv":
            self.post_conv = nn.Conv2d(
                in_channels=post_conv_channels,
                out_channels=post_conv_channels,
                kernel_size=3,
                padding=1,
                groups=post_conv_channels,
                bias=False,
            )
        elif self.post_conv_type == "conv":
            self.post_conv = nn.Conv2d(
                in_channels=post_conv_channels,
                out_channels=post_conv_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            )
        else:
            raise ValueError(
                f"Unknown post_conv_type: {self.post_conv_type}. "
                "Supported: ['none', 'dw_conv', 'conv']"
            )

        if self.use_multiscale:
            if multiscale_upsample_factor <= 0:
                raise ValueError(f"multiscale_upsample_factor must be > 0, got {multiscale_upsample_factor}")
            if self.patch_size % multiscale_upsample_factor != 0:
                raise ValueError(
                    f"patch_size ({self.patch_size}) must be divisible by "
                    f"multiscale_upsample_factor ({multiscale_upsample_factor})"
                )

            self.multiscale_upsample_factor = multiscale_upsample_factor
            self.multiscale_resize_type = multiscale_resize_type
            self.multiscale_merge_type = multiscale_merge_type
            self.multiscale_norm_type = multiscale_norm_type
            self.final_shuffle_factor = self.patch_size // self.multiscale_upsample_factor
            if self.multiscale_merge_type not in ("concat", "add_inplace"):
                raise ValueError(
                    f"Unknown multiscale_merge_type: {self.multiscale_merge_type}. "
                    "Supported: ['concat', 'add_inplace']"
                )
            if self.multiscale_norm_type not in ("shared", "per_level"):
                raise ValueError(
                    f"Unknown multiscale_norm_type: {self.multiscale_norm_type}. "
                    "Supported: ['shared', 'per_level']"
                )
            if self.multiscale_norm_type == "per_level":
                self.multiscale_norms = nn.ModuleList(
                    [nn.LayerNorm(dim_in, eps=eps) for _ in self.intermediate_layer_idx]
                )
            else:
                self.multiscale_norms = None

            if multiscale_out_channels is None:
                default_channels = max(dim_in // len(self.intermediate_layer_idx), 1)
                multiscale_out_channels = [default_channels] * len(self.intermediate_layer_idx)
            if len(multiscale_out_channels) != len(self.intermediate_layer_idx):
                raise ValueError(
                    "multiscale_out_channels length must match intermediate_layer_idx length. "
                    f"Got {len(multiscale_out_channels)} vs {len(self.intermediate_layer_idx)}"
                )

            self.projects = nn.ModuleList(
                [nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in multiscale_out_channels]
            )
            self.resize_layers = nn.ModuleList(
                [
                    build_multiscale_resize_layer(
                        out_channels=oc,
                        resize_type=self.multiscale_resize_type,
                        upsample_factor=self.multiscale_upsample_factor,
                    )
                    for oc in multiscale_out_channels
                ]
            )

            if self.multiscale_merge_type == "concat":
                fused_dim = sum(multiscale_out_channels)
            else:
                if len(set(multiscale_out_channels)) != 1:
                    raise ValueError(
                        "multiscale_out_channels must all be equal when multiscale_merge_type='add_inplace'. "
                        f"Got {multiscale_out_channels}"
                    )
                fused_dim = multiscale_out_channels[0]

            fused_dim = self._build_patch_embed_modules(
                fused_dim=fused_dim,
                dim_in=dim_in,
                eps=eps,
            )

            self.proj = self._build_multiscale_head(
                in_channels=fused_dim,
                out_channels=_output_dim * self.final_shuffle_factor ** 2,
                proj_type=proj_type,
                mlp_ratio=mlp_ratio,
                proj_bias=proj_bias,
            )
            self.proj_conf = self._build_multiscale_head(
                in_channels=fused_dim,
                out_channels=1 * self.final_shuffle_factor ** 2,
                proj_type=proj_type,
                mlp_ratio=mlp_ratio,
                proj_bias=proj_bias,
            )
            if self.predict_mask:
                self.proj_mask = self._build_multiscale_head(
                    in_channels=fused_dim,
                    out_channels=1 * self.final_shuffle_factor ** 2,
                    proj_type=proj_type,
                    mlp_ratio=mlp_ratio,
                    proj_bias=proj_bias,
                )
            else:
                self.proj_mask = None
        else:
            self.projects = None
            self.resize_layers = None
            self.multiscale_upsample_factor = None
            self.multiscale_resize_type = None
            self.multiscale_merge_type = None
            self.multiscale_norm_type = None
            self.multiscale_norms = None
            self.final_shuffle_factor = self.patch_size
            self._reset_patch_embed_modules()

            if proj_type == "linear":
                self.proj = nn.Linear(dim_in, _output_dim * self.patch_size ** 2, bias=proj_bias)
                self.proj_conf = nn.Linear(dim_in, 1 * self.patch_size ** 2, bias=proj_bias)
                if self.predict_mask:
                    self.proj_mask = nn.Linear(dim_in, 1 * self.patch_size ** 2, bias=proj_bias)
                else:
                    self.proj_mask = None
            elif proj_type == "mlp_self":
                hidden_dim = int(dim_in * mlp_ratio)
                self.proj = Mlp(dim_in, hidden_dim, _output_dim * self.patch_size ** 2, bias=proj_bias)
                self.proj_conf = Mlp(dim_in, hidden_dim, 1 * self.patch_size ** 2, bias=proj_bias)
                if self.predict_mask:
                    self.proj_mask = Mlp(dim_in, hidden_dim, 1 * self.patch_size ** 2, bias=proj_bias)
                else:
                    self.proj_mask = None
            else:
                raise ValueError(f"Unknown projection type: {proj_type}")

        if proj_zero_init:
            self._zero_init_last_layer(self.proj)
            self._zero_init_last_layer(self.proj_conf)
            if self.proj_mask is not None:
                self._zero_init_last_layer(self.proj_mask)

    def _validate_patch_embed_config(self, patch_embed_dim: Optional[int]) -> None:
        if self.patch_embed_merge_type not in ("concat", "add_inplace"):
            raise ValueError(
                f"Unknown patch_embed_merge_type: {self.patch_embed_merge_type}. "
                "Supported: ['concat', 'add_inplace']"
            )
        if self.use_patch_embed_intermediate and not self.use_multiscale:
            raise ValueError(
                "use_patch_embed_intermediate=True requires multiscale mode "
                "(set intermediate_layer_idx)."
            )
        if self.use_patch_embed_intermediate:
            if patch_embed_dim is None or patch_embed_dim <= 0:
                raise ValueError(
                    f"patch_embed_dim must be > 0 when use_patch_embed_intermediate=True, got {patch_embed_dim}"
                )
            if self.patch_embed_out_channels is not None and self.patch_embed_out_channels <= 0:
                raise ValueError(
                    "patch_embed_out_channels must be > 0 when provided for "
                    f"use_patch_embed_intermediate=True, got {self.patch_embed_out_channels}"
                )

    def _reset_patch_embed_modules(self) -> None:
        self.patch_embed_concat_channels = 0
        self.pe_norm = None
        self.pe_project = None
        self.patch_embed_resize_layer = None

    def _build_patch_embed_modules(
        self,
        fused_dim: int,
        dim_in: int,
        eps: float,
    ) -> int:
        if not self.use_patch_embed_intermediate:
            return fused_dim

        if self.patch_embed_out_channels is None:
            self.patch_embed_out_channels = max(dim_in // len(self.intermediate_layer_idx), 1)

        self.pe_norm = nn.LayerNorm(self.patch_embed_dim, eps=eps)
        self.pe_project = nn.Conv2d(
            in_channels=self.patch_embed_dim,
            out_channels=self.patch_embed_out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.patch_embed_resize_layer = build_multiscale_resize_layer(
            out_channels=self.patch_embed_out_channels,
            resize_type=self.multiscale_resize_type,
            upsample_factor=self.multiscale_upsample_factor,
        )

        if self.patch_embed_merge_type == "concat":
            self.patch_embed_concat_channels = self.patch_embed_out_channels
            return fused_dim + self.patch_embed_concat_channels

        if self.patch_embed_out_channels != fused_dim:
            raise ValueError(
                "patch_embed_out_channels must match fused multiscale channels when "
                "patch_embed_merge_type='add_inplace'. "
                f"Got patch_embed_out_channels={self.patch_embed_out_channels}, fused_dim={fused_dim}"
            )
        return fused_dim

    def _build_multiscale_head(
        self,
        in_channels: int,
        out_channels: int,
        proj_type: str,
        mlp_ratio: float,
        proj_bias: bool,
    ) -> nn.Module:
        def _build_spatial_conv(channels: int) -> Optional[nn.Module]:
            if self.multiscale_head_conv_type == "none":
                return None
            groups = channels if self.multiscale_head_conv_type == "dwconv" else 1
            return nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=groups,
                bias=False,
            )

        if proj_type == "linear":
            spatial_conv = _build_spatial_conv(in_channels)
            proj_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=proj_bias)
            if spatial_conv is None:
                return proj_conv
            return nn.Sequential(spatial_conv, proj_conv)

        if proj_type == "mlp_self":
            hidden_channels = int(in_channels * mlp_ratio)
            layers = [
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=proj_bias),
                nn.GELU(),
            ]
            spatial_conv = _build_spatial_conv(hidden_channels)
            if spatial_conv is not None:
                layers.append(spatial_conv)
            layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=proj_bias))
            return nn.Sequential(*layers)

        raise ValueError(f"Unknown projection type for multiscale mode: {proj_type}")

    def _zero_init_last_layer(self, proj: nn.Module) -> None:
        if isinstance(proj, nn.Linear):
            last_layer = proj
        elif isinstance(proj, Mlp):
            last_layer = proj.fc2
        elif isinstance(proj, nn.Conv2d):
            last_layer = proj
        elif isinstance(proj, nn.Sequential):
            last_layer = proj[-1]
        else:
            return

        nn.init.zeros_(last_layer.weight)
        if last_layer.bias is not None:
            nn.init.zeros_(last_layer.bias)

    def _build_pos_embed_map(
        self,
        channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return build_rope_like_pos_embed(
            height=height,
            width=width,
            channels=channels,
            pos_embed_type=self.pos_embed_type,
            base=self.pos_embed_base,
            dtype=dtype,
            device=device,
        ) * self.pos_embed_scale

    def _fuse_patch_embed_intermediate(
        self,
        fused_tokens: torch.Tensor,
        patch_embed_intermediate: Optional[List[torch.Tensor]],
        patch_h: int,
        patch_w: int,
    ) -> torch.Tensor:
        if not self.use_patch_embed_intermediate:
            return fused_tokens

        bs = fused_tokens.shape[0]
        target_h = fused_tokens.shape[-2]
        target_w = fused_tokens.shape[-1]

        if patch_embed_intermediate is None:
            raise ValueError("patch_embed_intermediate is required when use_patch_embed_intermediate=True")

        num_patches = patch_h * patch_w
        assert isinstance(patch_embed_intermediate, (list, tuple)), (
            "patch_embed_intermediate must be a list/tuple when "
            "use_patch_embed_intermediate=True."
        )
        assert len(patch_embed_intermediate) == 1, (
            "patch_embed_intermediate must contain exactly one tensor when "
            "use_patch_embed_intermediate=True."
        )

        layer_tokens = patch_embed_intermediate[0]
        if layer_tokens.dim() != 3:
            raise ValueError(
                "patch_embed_intermediate[0] must be [B*S, T, C]. "
                f"Got shape {tuple(layer_tokens.shape)}"
            )
        if layer_tokens.shape[0] != bs:
            raise ValueError(
                f"patch_embed_intermediate[0] batch mismatch: expected {bs}, got {layer_tokens.shape[0]}"
            )
        if layer_tokens.shape[1] < num_patches:
            raise ValueError(
                "patch_embed_intermediate[0] has too few tokens: "
                f"{layer_tokens.shape[1]} < num_patches({num_patches})"
            )
        if layer_tokens.shape[-1] != self.patch_embed_dim:
            raise ValueError(
                f"patch_embed_intermediate[0] channel mismatch: expected {self.patch_embed_dim}, "
                f"got {layer_tokens.shape[-1]}"
            )

        # DINO intermediates include cls/storage prefixes. Taking the tail patch tokens
        # is robust to different numbers of prefix tokens.
        layer_tokens = layer_tokens[:, -num_patches:, :]
        if self.disable_last_layer_amp:
            layer_tokens = layer_tokens.float()
        layer_tokens = self.pe_norm(layer_tokens)
        layer_tokens = layer_tokens.permute(0, 2, 1).reshape(bs, -1, patch_h, patch_w)
        layer_tokens = self.pe_project(layer_tokens)
        layer_tokens = self.patch_embed_resize_layer(layer_tokens)
        if layer_tokens.shape[-2:] != (target_h, target_w):
            raise ValueError(
                "patch_embed_intermediate branch spatial mismatch: "
                f"expected {(target_h, target_w)}, got {tuple(layer_tokens.shape[-2:])}"
            )

        if self.patch_embed_merge_type == "concat":
            return torch.cat([fused_tokens, layer_tokens], dim=1)
        return fused_tokens + layer_tokens

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = -1,
        patch_embed_intermediate: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the VGG Linear head.
        
        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.

        Returns:
            Tuple[Tensor, Tensor, Optional[Tensor]]: Tuple of (predictions, confidence, mask).
        """
        B, S, _, H, W = images.shape
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        last_layer_amp_context = (
            torch.cuda.amp.autocast(enabled=False)
            if self.disable_last_layer_amp
            else contextlib.nullcontext()
        )
        
        with last_layer_amp_context:
            if self.use_multiscale:
                features = []
                fused_tokens = None
                for dpt_idx, layer_idx in enumerate(self.intermediate_layer_idx):
                    layer_tokens = aggregated_tokens_list[layer_idx]
                    _, _, T, C = layer_tokens.shape
                    layer_tokens = layer_tokens.view(B * S, T, C)
                    layer_tokens = layer_tokens[:, patch_start_idx:]
                    if self.disable_last_layer_amp:
                        layer_tokens = layer_tokens.float()
                    if self.multiscale_norm_type == "per_level":
                        layer_tokens = self.multiscale_norms[dpt_idx](layer_tokens)
                    else:
                        layer_tokens = self.norm(layer_tokens)
                    layer_tokens = layer_tokens.permute(0, 2, 1).reshape(B * S, -1, patch_h, patch_w)
                    layer_tokens = self.projects[dpt_idx](layer_tokens)
                    layer_tokens = self.resize_layers[dpt_idx](layer_tokens)
                    if self.multiscale_merge_type == "concat":
                        features.append(layer_tokens)
                    else:
                        if fused_tokens is None:
                            fused_tokens = layer_tokens
                        else:
                            fused_tokens = fused_tokens + layer_tokens

                if self.multiscale_merge_type == "concat":
                    fused_tokens = torch.cat(features, dim=1)

                if self.use_patch_embed_intermediate:
                    fused_tokens = self._fuse_patch_embed_intermediate(
                        fused_tokens=fused_tokens,
                        patch_embed_intermediate=patch_embed_intermediate,
                        patch_h=patch_h,
                        patch_w=patch_w,
                    )

                if self.pos_embed_type != "none":
                    fused_tokens = fused_tokens + self._build_pos_embed_map(
                        channels=fused_tokens.shape[1],
                        height=fused_tokens.shape[-2],
                        width=fused_tokens.shape[-1],
                        dtype=fused_tokens.dtype,
                        device=fused_tokens.device,
                    )

                feat = self.proj(fused_tokens)
                feat_conf = self.proj_conf(fused_tokens)
                if self.predict_mask:
                    feat_mask = self.proj_mask(fused_tokens)
            else:
                # Legacy path: use only the last layer.
                tokens = aggregated_tokens_list[-1]
                _, _, T, C = tokens.shape
                tokens = tokens.view(B * S, T, C)
                tokens = tokens[:, patch_start_idx:]

                if self.disable_last_layer_amp:
                    tokens = tokens.float()
                tokens = self.norm(tokens)
                if self.pos_embed_type != "none":
                    tokens_2d = tokens.permute(0, 2, 1).reshape(B * S, -1, patch_h, patch_w)
                    tokens_2d = tokens_2d + self._build_pos_embed_map(
                        channels=tokens_2d.shape[1],
                        height=tokens_2d.shape[2],
                        width=tokens_2d.shape[3],
                        dtype=tokens_2d.dtype,
                        device=tokens_2d.device,
                    )
                    tokens = tokens_2d.reshape(B * S, -1, patch_h * patch_w).permute(0, 2, 1).contiguous()

                feat = self.proj(tokens)
                feat_conf = self.proj_conf(tokens)
                if self.predict_mask:
                    feat_mask = self.proj_mask(tokens)

                feat = feat.transpose(-1, -2).contiguous().view(B * S, -1, patch_h, patch_w)
                feat_conf = feat_conf.transpose(-1, -2).contiguous().view(B * S, -1, patch_h, patch_w)
                if self.predict_mask:
                    feat_mask = feat_mask.transpose(-1, -2).contiguous().view(B * S, -1, patch_h, patch_w)

            if self.final_shuffle_factor > 1:
                feat = F.pixel_shuffle(feat, self.final_shuffle_factor)
                feat_conf = F.pixel_shuffle(feat_conf, self.final_shuffle_factor)
                if self.predict_mask:
                    feat_mask = F.pixel_shuffle(feat_mask, self.final_shuffle_factor)

            if self.post_conv is not None:
                cat_tensors = [feat, feat_conf]
                if self.predict_mask:
                    cat_tensors.append(feat_mask)

                pred_channels = feat.shape[1]
                feat_post = torch.cat(cat_tensors, dim=1).contiguous()
                feat_post = self.post_conv(feat_post)

                feat = feat_post[:, :pred_channels]
                feat_conf = feat_post[:, pred_channels : pred_channels + 1]
                if self.predict_mask:
                    feat_mask = feat_post[:, pred_channels + 1 : pred_channels + 2]

            if self.activation == "none":
                preds = feat.permute(0, 2, 3, 1) # [B*S, H, W, C]
                conf = feat_conf.permute(0, 2, 3, 1)
            else:
                # Concatenate pred features with conf features for activate_head
                feat_combined = torch.cat([feat, feat_conf], dim=1)
                preds, conf = activate_head(feat_combined, activation=self.activation, conf_activation=self.conf_activation)
                
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
                f"VGGLinearHead outputs must be fp32, got preds={preds.dtype}, conf={conf.dtype}"
            )

            return preds, conf, mask
