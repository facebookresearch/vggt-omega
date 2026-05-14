# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib

import torch
import torch.nn as nn
from typing import Optional, Union, List

from vggt_omega.models.layers import SelfAttentionBlock
from vggt_omega.models.heads.head_act import base_pose_act


class CameraHeadLinear(nn.Module):
    """
    CameraHeadLinear predicts camera parameters from token representations using a simple MLP/Linear layer.
    It removes the transformer trunk and iterative refinement found in CameraHead.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        pose_encoding_type: str = "absT_quaR_FoV",
        mlp_ratio: Union[int, float, List[float]] = [0.5],  # Can be a single ratio or a list of ratios for multiple hidden layers
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",
        eps: float = 1e-5,
        # Extra attention blocks
        extra_attention_depth: int = 4,
        extra_attention_dim: int = -1,  # If < 0, skip projection and keep dim_in
        extra_attention_num_heads: int = 16,
        extra_attention_mlp_ratio: float = 4.0,
        extra_attention_qkv_bias: bool = True,
        extra_attention_proj_bias: bool = True,
        extra_attention_ffn_bias: bool = True,
        extra_attention_use_qk_norm: bool = False,
        extra_attention_use_all_special_tokens: bool = True,  # If False, only feed pose token to extra attention
        extra_attention_init_values: float = 1e-5,
        extra_attention_mask_k_bias: bool = True,
        extra_attention_post_norm: bool = True,  # Whether to apply LayerNorm after extra attention blocks
        extra_attention_pre_norm: bool = True,  # Whether to apply LayerNorm before extra attention blocks
        patch_size: int = 16,
        disable_last_layer_amp: bool = True,
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.disable_last_layer_amp = disable_last_layer_amp

        if pose_encoding_type in ("absT_quaR_FoV", "absT_quaR_FoV_c2w"):
            self.target_dim = 9
        elif pose_encoding_type == "absT_quaR_FoV_PP":
            # self.target_dim = 11
            raise NotImplementedError(
                "YOU NEED TO DOUBLE CHECK THE TARGET DIMENSION FOR THE PP VARIANT."
            )
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.pose_encoding_type = pose_encoding_type
        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.extra_attention_use_qk_norm = extra_attention_use_qk_norm
        self.extra_attention_use_all_special_tokens = extra_attention_use_all_special_tokens
        self.extra_attention_pre_norm_enabled = extra_attention_pre_norm

        # Build extra attention blocks if enabled
        if extra_attention_depth > 0:
            attention_dim = dim_in if extra_attention_dim < 0 else extra_attention_dim
            self.extra_attention_dim = attention_dim
            self.extra_attention_pre_norm = (
                nn.LayerNorm(dim_in, eps=eps)
                if extra_attention_pre_norm
                else nn.Identity()
            )
            # Project from dim_in to attention_dim unless disabled by extra_attention_dim < 0.
            self.extra_attention_in_proj = (
                nn.Identity()
                if extra_attention_dim < 0
                else nn.Linear(dim_in, attention_dim)
            )
            self.extra_attention_blocks = nn.ModuleList([
                SelfAttentionBlock(
                    dim=attention_dim,
                    num_heads=extra_attention_num_heads,
                    ffn_ratio=extra_attention_mlp_ratio,
                    qkv_bias=extra_attention_qkv_bias,
                    proj_bias=extra_attention_proj_bias,
                    ffn_bias=extra_attention_ffn_bias,
                    init_values=extra_attention_init_values,
                    use_qk_norm=extra_attention_use_qk_norm,
                    mask_k_bias=extra_attention_mask_k_bias,
                ) for _ in range(extra_attention_depth)
            ])
            
            # Feature dimension after extra attention
            feature_dim = attention_dim
        else:
            self.extra_attention_dim = None
            self.extra_attention_pre_norm = None
            self.extra_attention_in_proj = None
            self.extra_attention_blocks = None
            feature_dim = dim_in

        # Normalization for camera token
        # Matches LinearHead's behavior: controlled by extra_attention_post_norm
        if extra_attention_post_norm:
            self.token_norm = nn.LayerNorm(feature_dim, eps=eps)
        else:
            self.token_norm = nn.Identity()

        # Projection head (MLP)
        if isinstance(mlp_ratio, (float, int)):
            mlp_ratio = [mlp_ratio]

        layers = []
        current_dim = feature_dim
        for ratio in mlp_ratio:
            hidden_dim = int(feature_dim * ratio)
            layers.append(nn.Linear(current_dim, hidden_dim, bias=True))
            layers.append(nn.GELU())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, self.target_dim, bias=True))

        
        self.pose_branch = nn.Sequential(*layers)
        
    def forward(
        self,
        aggregated_tokens_list: list,
        images: torch.Tensor = None,
        patch_start_idx: Optional[int] = None,
    ) -> list:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            images (Tensor, optional): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int, optional): Starting index for patch tokens.
                Required when extra attention is enabled and all special tokens are used.
        Returns:
            list: A list containing the predicted camera encodings (post-activation).
                  Returned as a list to maintain interface compatibility with CameraHead.
        """
        # Use tokens from the last block for camera prediction.
        tokens = aggregated_tokens_list[-1]
        B, S, T, C = tokens.shape
        

        # Apply extra attention if enabled
        if self.extra_attention_blocks is not None:
            if self.extra_attention_use_all_special_tokens:
                if patch_start_idx is None:
                    raise ValueError(
                        "patch_start_idx is required when extra_attention_depth > 0 "
                        "and extra_attention_use_all_special_tokens=True"
                    )
                if patch_start_idx > T:
                    raise ValueError(
                        f"patch_start_idx ({patch_start_idx}) exceeds token length ({T})"
                    )
                num_special_tokens = patch_start_idx
            else:
                num_special_tokens = 1

            if self.disable_last_layer_amp:
                if tokens.dtype != torch.float32:
                    tokens = tokens.float()
                    
            # Extract special tokens only and project to extra_attention_dim
            special_tokens = tokens[:, :, :num_special_tokens, :]
            special_tokens = self.extra_attention_pre_norm(special_tokens)
            special_tokens = self.extra_attention_in_proj(special_tokens)

            # Apply global attention over special tokens across all frames.
            # RoPE is not used here since special tokens do not have spatial positions.
            special_tokens = special_tokens.reshape(B, S * num_special_tokens, -1)
            rope_sincos = None
            for blk in self.extra_attention_blocks:
                special_tokens = blk(special_tokens, rope_sincos)

            # Restore per-frame layout and take the camera token
            special_tokens = special_tokens.reshape(B, S, num_special_tokens, -1)
            pose_tokens = special_tokens[:, :, 0]
        else:
            # Extract the camera tokens (index 0)
            # tokens shape: [B, S, P, C] -> pose_tokens: [B, S, C]
            pose_tokens = tokens[:, :, 0]

        # Apply normalization
        # pose_tokens = self.token_norm(pose_tokens)

        last_layer_amp_context = (
            torch.autocast(device_type="cuda", enabled=False)
            if self.disable_last_layer_amp
            else contextlib.nullcontext()
        )

        with last_layer_amp_context:
            if self.disable_last_layer_amp:
                if pose_tokens.dtype != torch.float32:
                    pose_tokens = pose_tokens.float()
            
            pose_tokens = self.token_norm(pose_tokens)
            
            # Predict pose encoding
            pred_pose_enc = self.pose_branch(pose_tokens)

            # Apply activations
            activated_pose = self.apply_pose_activation(pred_pose_enc)

        # Return as a list
        return [activated_pose]

    def apply_pose_activation(self, pred_pose_enc: torch.Tensor) -> torch.Tensor:
        """
        Apply activations to pose encoding.
        """
        if self.pose_encoding_type in ("absT_quaR_FoV", "absT_quaR_FoV_c2w"):
            T = pred_pose_enc[..., :3]
            quat = pred_pose_enc[..., 3:7]
            fl = pred_pose_enc[..., 7:]  # or fov

            T = base_pose_act(T, self.trans_act)
            quat = base_pose_act(quat, self.quat_act)
            fl = base_pose_act(fl, self.fl_act)  # or fov
            fl = fl + 0.01  # for stability

            pred_pose_enc = torch.cat([T, quat, fl], dim=-1)
            return pred_pose_enc
        raise ValueError(f"Unsupported camera encoding type: {self.pose_encoding_type}")
