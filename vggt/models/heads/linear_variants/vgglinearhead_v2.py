from typing import List, Optional, Tuple

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_fusion_adapter import (
    DPTFusionAdapter,
)
from .gated_progressive_fusion_adapter import GatedProgressiveFusionAdapter
from ..head_act import activate_head
from .head_config_checks import (
    build_fusion_input_channels_or_raise,
    get_final_shuffle_factor_or_raise,
    resolve_multiscale_out_channels_or_raise,
    resolve_patch_embed_out_channels_or_raise,
    validate_head_core_config_or_raise,
)
from .head_utils_v2 import (
    build_post_conv_or_raise,
    build_prediction_head,
    tokens_to_2d_feature_map,
)
from .token_shape_checks import (
    get_aggregated_layer_or_raise,
    get_aggregated_patch_tokens_or_raise,
    get_image_patch_grid_or_raise,
    get_patch_embed_tokens_or_raise,
)


class VGGLinearHeadV2(nn.Module):
    """
    Stage-1 refactor:
    - normalize + project token branches
    - build an ordered feature list:
      [patch_embed_intermediate[0] (if used), aggregated_tokens_list[idx1], ...]

    Later stages can consume this ordered list for resize/merge/head prediction.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 16,
        output_dim: int = 4,
        activation: str = "norm_exp",
        conf_activation: str = "softplus_0.5_10",
        predict_mask: bool = True,
        mask_activation: str = "none",
        proj_type: str = "linear",
        mlp_ratio: float = 2.0,
        proj_bias: bool = True,
        intermediate_layer_idx: Optional[List[int]] = None,
        multiscale_out_channels: Optional[List[int]] = None,
        multiscale_norm_type: str = "per_level",
        use_patch_embed_intermediate: bool = False,
        patch_embed_dim: Optional[int] = None,
        patch_embed_out_channels: Optional[int] = None,
        post_conv_mode: str = "none",  # Supported: ["none", "res3x3", "res5x5", "res2_3x3_gelu"].
        eps: float = 1e-5,
        disable_last_layer_amp: bool = True,
        fusion_mode: str = "gated_progressive",  # Controls fusion in forward(); currently supports: ["dpt", "gated_progressive"].
        dpt_fusion_features: int = 256,  # Output channel width for DPT fusion adapter.
        dpt_use_interpolate_conv2d_upsample: bool = False,  # If True, replace DPT ConvTranspose2d upsample with interpolate + Conv2d.
        dpt_align_corners: bool = True,  # Controls align_corners for bilinear upsample ops in DPT fusion.
        use_identity_resconfunit1: bool = False,  # If True (DPT mode), set all refinenet resConfUnit1 blocks to Identity.
        fuser_output_scale: int = 4,  # Supported fused feature scales: 4 (1/4 output) or 1 (patch-grid output) in gated mode.
        gated_progressive_align_channels: bool = True,  # Align input channels via 1x1 convs before gated fusion (applies to both single-branch and 4-branch paths).
        gated_progressive_fusion_channels: int = 256,  # Target channel width used by gated fusion when alignment is enabled.
        gated_progressive_group_norm_groups: int = 1,  # GroupNorm group count in gated fusion blocks.
        gated_progressive_two_stage_upsample: bool = False,  # If True, upsample 1/16->1/4 as two 2x steps instead of one 4x step.
        gated_progressive_upsample_use_pointwise_conv: bool = True,  # Apply 1x1 pointwise conv after upsample repair depthwise block.
        gated_progressive_rcu_use_group_norm: bool = False,  # Enable GroupNorm(1, C) in ResidualConvUnit blocks.
        **kwargs,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.output_dim = output_dim
        self.activation = activation
        self.conf_activation = conf_activation
        self.predict_mask = predict_mask
        self.mask_activation = mask_activation
        self.proj_type = proj_type
        self.disable_last_layer_amp = disable_last_layer_amp
        self.post_conv_mode = post_conv_mode
        if kwargs:
            unexpected_keys = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected_keys}")
        self.fusion_mode = fusion_mode
        self.dpt_fusion_features = dpt_fusion_features
        self.dpt_use_interpolate_conv2d_upsample = dpt_use_interpolate_conv2d_upsample
        self.dpt_align_corners = dpt_align_corners
        assert use_identity_resconfunit1 == False, "use_identity_resconfunit1 must be False"
        self.use_identity_resconfunit1 = use_identity_resconfunit1
        self.fuser_output_scale = fuser_output_scale
        self.intermediate_layer_idx = intermediate_layer_idx or []
        self.multiscale_norm_type = multiscale_norm_type
        self.use_patch_embed_intermediate = use_patch_embed_intermediate
        self.patch_embed_dim = patch_embed_dim
        self.patch_embed_out_channels = patch_embed_out_channels
        self.multiscale_out_channels = multiscale_out_channels

        validate_head_core_config_or_raise(
            intermediate_layer_idx=self.intermediate_layer_idx,
            use_patch_embed_intermediate=self.use_patch_embed_intermediate,
            multiscale_norm_type=self.multiscale_norm_type,
            fusion_mode=self.fusion_mode,
            dpt_fusion_features=self.dpt_fusion_features,
            fuser_output_scale=self.fuser_output_scale,
        )

        self.gated_progressive_align_channels = gated_progressive_align_channels
        self.gated_progressive_fusion_channels = gated_progressive_fusion_channels
        self.gated_progressive_group_norm_groups = gated_progressive_group_norm_groups
        self.gated_progressive_rcu_use_group_norm = gated_progressive_rcu_use_group_norm
        self.gated_progressive_two_stage_upsample = gated_progressive_two_stage_upsample
        self.gated_progressive_upsample_use_pointwise_conv = (
            gated_progressive_upsample_use_pointwise_conv
        )

        self.multiscale_out_channels = resolve_multiscale_out_channels_or_raise(
            multiscale_out_channels=self.multiscale_out_channels,
            dim_in=dim_in,
            num_levels=len(self.intermediate_layer_idx),
        )

        if self.multiscale_norm_type == "per_level":
            self.norm = nn.Identity()
            self.multiscale_norms = nn.ModuleList(
                [nn.LayerNorm(dim_in, eps=eps) for _ in self.intermediate_layer_idx]
            )
        else:
            self.norm = nn.LayerNorm(dim_in, eps=eps)
            self.multiscale_norms = None

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=proj_bias,
                )
                for out_channels in self.multiscale_out_channels
            ]
        )

        if self.use_patch_embed_intermediate:
            self.patch_embed_out_channels = resolve_patch_embed_out_channels_or_raise(
                use_patch_embed_intermediate=self.use_patch_embed_intermediate,
                patch_embed_dim=self.patch_embed_dim,
                patch_embed_out_channels=self.patch_embed_out_channels,
                dim_in=dim_in,
                num_aggregated_levels=len(self.intermediate_layer_idx),
            )
            self.pe_norm = nn.LayerNorm(self.patch_embed_dim, eps=eps)
            self.pe_project = nn.Conv2d(
                in_channels=self.patch_embed_dim,
                out_channels=self.patch_embed_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=proj_bias,
            )
        else:
            self.pe_norm = None
            self.pe_project = None

        self.fusion_input_channels = build_fusion_input_channels_or_raise(
            multiscale_out_channels=self.multiscale_out_channels,
            use_patch_embed_intermediate=self.use_patch_embed_intermediate,
            patch_embed_out_channels=self.patch_embed_out_channels,
            fusion_mode=self.fusion_mode,
        )

        if self.fusion_mode == "dpt":
            self.fuser = DPTFusionAdapter(
                out_channels=self.fusion_input_channels,
                fusion_features=self.dpt_fusion_features,
                use_interpolate_conv2d_upsample=self.dpt_use_interpolate_conv2d_upsample,
                align_corners=self.dpt_align_corners,
                use_identity_resconfunit1=self.use_identity_resconfunit1,
            )
            fused_dim = self.dpt_fusion_features
        else:
            self.fuser = GatedProgressiveFusionAdapter(
                out_channels=self.fusion_input_channels,
                align_channels=self.gated_progressive_align_channels,
                fusion_channels=self.gated_progressive_fusion_channels,
                rcu_use_group_norm=self.gated_progressive_rcu_use_group_norm,
                two_stage_upsample=self.gated_progressive_two_stage_upsample,
                upsample_use_pointwise_conv=self.gated_progressive_upsample_use_pointwise_conv,
                include_upsample=(self.fuser_output_scale == 4),
            )
            if self.gated_progressive_align_channels:
                fused_dim = self.gated_progressive_fusion_channels
            else:
                fused_dim = self.fusion_input_channels[0]

        self.final_shuffle_factor = get_final_shuffle_factor_or_raise(
            output_dim=self.output_dim,
            patch_size=self.patch_size,
            fuser_output_scale=self.fuser_output_scale,
        )

        pred_channels = self.output_dim - 1
        self.proj = build_prediction_head(
            in_channels=fused_dim,
            out_channels=pred_channels * self.final_shuffle_factor ** 2,
            proj_type=self.proj_type,
            mlp_ratio=mlp_ratio,
            proj_bias=proj_bias,
        )
        self.proj_conf = build_prediction_head(
            in_channels=fused_dim,
            out_channels=self.final_shuffle_factor ** 2,
            proj_type=self.proj_type,
            mlp_ratio=mlp_ratio,
            proj_bias=proj_bias,
        )
        if self.predict_mask:
            self.proj_mask = build_prediction_head(
                in_channels=fused_dim,
                out_channels=self.final_shuffle_factor ** 2,
                proj_type=self.proj_type,
                mlp_ratio=mlp_ratio,
                proj_bias=proj_bias,
            )
        else:
            self.proj_mask = None

        # Post-conv is only applied to regression + confidence branches.
        post_conv_channels = pred_channels + 1
        self.post_conv = build_post_conv_or_raise(
            post_conv_mode=self.post_conv_mode,
            post_conv_channels=post_conv_channels,
        )
        


    def _project_aggregated_branch(
        self,
        layer_tokens: torch.Tensor,
        branch_idx: int,
        patch_start_idx: int,
        num_patches: int,
        patch_h: int,
        patch_w: int,
        expected_bs_flat: Optional[int] = None,
    ) -> torch.Tensor:
        patch_tokens, bs_flat = get_aggregated_patch_tokens_or_raise(
            layer_tokens=layer_tokens,
            patch_start_idx=patch_start_idx,
            num_patches=num_patches,
            expected_bs_flat=expected_bs_flat,
        )

        if self.disable_last_layer_amp:
            if patch_tokens.dtype != torch.float32:
                patch_tokens = patch_tokens.float()

        if self.multiscale_norm_type == "per_level":
            patch_tokens = self.multiscale_norms[branch_idx](patch_tokens)
        else:
            patch_tokens = self.norm(patch_tokens)

        feature_map = tokens_to_2d_feature_map(
            tokens=patch_tokens,
            bs_flat=bs_flat,
            patch_h=patch_h,
            patch_w=patch_w,
        )
        return self.projects[branch_idx](feature_map)

    def _project_patch_embed_branch(
        self,
        patch_embed_intermediate: Optional[List[torch.Tensor]],
        bs_flat: int,
        num_patches: int,
        patch_h: int,
        patch_w: int,
    ) -> Optional[torch.Tensor]:
        if not self.use_patch_embed_intermediate:
            return None
        layer_tokens = get_patch_embed_tokens_or_raise(
            patch_embed_intermediate=patch_embed_intermediate,
            expected_bs_flat=bs_flat,
            expected_num_patches=num_patches,
            expected_channels=self.patch_embed_dim,
        )

        # Keep the tail patch tokens to be robust to variable prefix-token counts.
        layer_tokens = layer_tokens[:, -num_patches:, :]
        if self.disable_last_layer_amp:
            if layer_tokens.dtype != torch.float32:
                layer_tokens = layer_tokens.float()
        layer_tokens = self.pe_norm(layer_tokens)
        feature_map = tokens_to_2d_feature_map(
            tokens=layer_tokens,
            bs_flat=bs_flat,
            patch_h=patch_h,
            patch_w=patch_w,
        )
        return self.pe_project(feature_map)

    def build_norm_projected_token_list(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        patch_embed_intermediate: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        bsz, seq_len, patch_h, patch_w, num_patches = get_image_patch_grid_or_raise(
            images=images,
            patch_size=self.patch_size,
        )
        bs_flat = bsz * seq_len

        projected_tokens: List[torch.Tensor] = []

        patch_embed_feature = self._project_patch_embed_branch(
            patch_embed_intermediate=patch_embed_intermediate,
            bs_flat=bs_flat,
            num_patches=num_patches,
            patch_h=patch_h,
            patch_w=patch_w,
        )
        if patch_embed_feature is not None:
            projected_tokens.append(patch_embed_feature)

        for branch_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            layer_tokens = get_aggregated_layer_or_raise(
                aggregated_tokens_list=aggregated_tokens_list,
                layer_idx=layer_idx,
            )
            projected_feature = self._project_aggregated_branch(
                layer_tokens=layer_tokens,
                branch_idx=branch_idx,
                patch_start_idx=patch_start_idx,
                num_patches=num_patches,
                patch_h=patch_h,
                patch_w=patch_w,
                expected_bs_flat=bs_flat,
            )
            projected_tokens.append(projected_feature)
        return projected_tokens

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        patch_embed_intermediate: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len = images.shape[:2]
        last_layer_amp_context = (
            torch.cuda.amp.autocast(enabled=False)
            if self.disable_last_layer_amp
            else contextlib.nullcontext()
        )

        with last_layer_amp_context:
            projected_tokens = self.build_norm_projected_token_list(
                aggregated_tokens_list=aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                patch_embed_intermediate=patch_embed_intermediate,
            )
            if self.fusion_mode == "gated_progressive":
                return_quarter = self.fuser_output_scale == 4
                fused_tokens = self.fuser(
                    projected_tokens,
                    return_quarter=return_quarter,
                )
            else:
                fused_tokens = self.fuser(projected_tokens)
            
            feat = self.proj(fused_tokens)
            feat_conf = self.proj_conf(fused_tokens)
            if self.predict_mask:
                feat_mask = self.proj_mask(fused_tokens)

            if self.final_shuffle_factor > 1:
                feat = F.pixel_shuffle(feat, self.final_shuffle_factor)
                feat_conf = F.pixel_shuffle(feat_conf, self.final_shuffle_factor)
                if self.predict_mask:
                    feat_mask = F.pixel_shuffle(feat_mask, self.final_shuffle_factor)

            if self.post_conv is not None:
                pred_channels = feat.shape[1]
                feat_cat = torch.cat([feat, feat_conf], dim=1).contiguous()
                feat_post = self.post_conv(feat_cat)

                feat = feat_post[:, :pred_channels]
                feat_conf = feat_post[:, pred_channels : pred_channels + 1]

            if self.activation == "none":
                preds = feat.permute(0, 2, 3, 1)
                conf = feat_conf.permute(0, 2, 3, 1)
            else:
                feat_combined = torch.cat([feat, feat_conf], dim=1)
                preds, conf = activate_head(
                    feat_combined,
                    activation=self.activation,
                    conf_activation=self.conf_activation,
                )

            if self.predict_mask:
                if self.mask_activation == "sigmoid":
                    mask = torch.sigmoid(feat_mask)
                elif self.mask_activation == "none":
                    mask = feat_mask
                else:
                    raise ValueError(f"Unknown mask activation: {self.mask_activation}")
            else:
                mask = None

            preds = preds.view(bsz, seq_len, *preds.shape[1:])
            conf = conf.view(bsz, seq_len, *conf.shape[1:])
            if mask is not None:
                mask = mask.view(bsz, seq_len, *mask.shape[1:])

            # assert preds.dtype == torch.float32 and conf.dtype == torch.float32, (
            #     f"VGGLinearHead outputs must be fp32, got preds={preds.dtype}, conf={conf.dtype}"
            # )

            return preds, conf, mask

