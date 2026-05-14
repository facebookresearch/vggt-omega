# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.models.layers.block import SelfAttentionBlock
from vggt.models.layers.ffn_layers import Mlp

from vggt.models.heads.head_act import base_pose_act


class CameraHead(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 1e-5,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
        linear_forward: bool = False,
        use_qk_norm: bool = False,
        embed_pose_type: str = "linear",
        use_gate_msa: bool = True,
        use_empty_pose_tokens: bool = True,
        pose_branch_camera_init: bool = False,
        pose_branch_init_fov_bias: Union[float, List[float]] = 1.0,
        pose_branch_init_last_layer_std: float = 1e-3,
        eps = 1e-5,
    ):
        super().__init__()
        
        # What we can try:
        #  [skipped] 1. Init self.empty_pose_tokens with random values; How to init the empty pose tokens?
        #  [x]  2. self.embed_pose to stronger arch such as a three-layer MLP
        #    3. self.pose_branch, MLP or simple Linear layer?
        #    4. Swiglu FFN layer
        #  [x] 5. Camera from world or camera to world?
        #    6. Uncertaincty Camera Head
        #  [x]  7. Predict extrinsic and intrinsic separately? decoupled camera head
        #  [skipped]  8. Would it be helpful to also involve the registers? 
        #    9. Refine in the latent space of camera tokens? (note currently the pose tokens are not updated in every iteration.)
        #  [x]  weight_trans  10. Bigger weights on translation? 
        #  [x]  11. Remove empty_pose_tokens? How about just using the pose tokens in the first step directly?
        #  [x] 12. poseLN_modulation Remove gate_msa?

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        elif pose_encoding_type == "absT_quaR_FoV_PP":
            raise NotImplementedError("YOU NEED TO DOUBLE CHECK THE TARGET DIMENSION FOR THE PP VARIANT.")
            self.target_dim = 11
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.pose_encoding_type = pose_encoding_type
        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth
        self.linear_forward = linear_forward
        self.use_gate_msa = use_gate_msa
        self.use_empty_pose_tokens = use_empty_pose_tokens
        self.pose_branch_camera_init = pose_branch_camera_init
        self.pose_branch_init_fov_bias = pose_branch_init_fov_bias
        self.pose_branch_init_last_layer_std = pose_branch_init_last_layer_std

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                SelfAttentionBlock(dim=dim_in, num_heads=num_heads, ffn_ratio=mlp_ratio, init_values=init_values, use_qk_norm=use_qk_norm)
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in, eps=eps)
        self.trunk_norm = nn.LayerNorm(dim_in, eps=eps)

        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)
        if self.pose_branch_camera_init:
            self._init_pose_branch_with_camera_prior()

        if not self.linear_forward: 
            if self.use_empty_pose_tokens:
                # Learnable empty camera pose token.
                self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
            
            if embed_pose_type == "linear":
                self.embed_pose = nn.Linear(self.target_dim, dim_in)
            elif embed_pose_type == "mlp":
                self.embed_pose = Mlp(in_features=self.target_dim, hidden_features=dim_in // 2, out_features=dim_in, drop=0)
            else:
                raise ValueError(f"Unsupported embed pose type: {embed_pose_type}")

            # Module for producing modulation parameters: shift, scale, and a gate.
            if self.use_gate_msa:
                modulation_dim = 3 * dim_in
            else:
                modulation_dim = 2 * dim_in
            self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, modulation_dim, bias=True))

            # Adaptive layer normalization without affine parameters.
            self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=eps)

    def _init_pose_branch_with_camera_prior(self) -> None:
        linear_layers = [m for m in self.pose_branch.modules() if isinstance(m, nn.Linear)]
        if not linear_layers:
            return

        for layer in linear_layers[:-1]:
            nn.init.trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        last = linear_layers[-1]
        nn.init.normal_(last.weight, std=self.pose_branch_init_last_layer_std)
        if last.bias is not None:
            nn.init.zeros_(last.bias)
            with torch.no_grad():
                # Quaternion is xyzw in this codebase, so identity rotation is w=1.
                last.bias[6] = 1.0
                if isinstance(self.pose_branch_init_fov_bias, (list, tuple)):
                    if len(self.pose_branch_init_fov_bias) != 2:
                        raise ValueError(
                            "pose_branch_init_fov_bias must be a float or a length-2 sequence"
                        )
                    fov_h, fov_w = self.pose_branch_init_fov_bias
                else:
                    fov_h = fov_w = self.pose_branch_init_fov_bias
                last.bias[7] = fov_h
                last.bias[8] = fov_w


    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4) -> list:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.

        Returns:
            list: A list of predicted camera encodings (post-activation) from each iteration.
        """        
        # Use tokens from the last block for camera prediction.
        tokens = aggregated_tokens_list[-1]
        
        # Extract the camera tokens
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)

        if self.linear_forward:
            pred_pose_enc_list = self.linear_forward_fn(pose_tokens, num_iterations)
        else:
            pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations)
        return pred_pose_enc_list
    
    def linear_forward_fn(self, pose_tokens: torch.Tensor, num_iterations: int) -> list:
        """
        Linear forward pass to predict camera parameters.
        """
        B, S, C = pose_tokens.shape 
        pred_pose_enc = None
        pred_pose_enc_list = []
        
        pose_tokens_modulated = self.trunk(pose_tokens)
        pred_pose_enc = self.pose_branch(self.trunk_norm(pose_tokens_modulated))
        activated_pose = self.apply_pose_activation(pred_pose_enc)
        pred_pose_enc_list.append(activated_pose)
        return pred_pose_enc_list



    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int) -> list:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, 1, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of activated camera encodings from each iteration.
        """
        B, S, C = pose_tokens.shape  # S is expected to be 1.
        pred_pose_enc = None
        pred_pose_enc_list = []

        if not self.use_empty_pose_tokens:
            # The first prediction is not iterative.
            pose_tokens_modulated = self.trunk(pose_tokens)
            pred_pose_enc = self.pose_branch(self.trunk_norm(pose_tokens_modulated))
            activated_pose = self.apply_pose_activation(pred_pose_enc)
            pred_pose_enc_list.append(activated_pose)
            loop_range = range(num_iterations - 1)
        else:
            loop_range = range(num_iterations)

        for _ in loop_range:
            # Use a learned empty pose for the first iteration.
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            mod_params = self.poseLN_modulation(module_input)
            if self.use_gate_msa:
                shift_msa, scale_msa, gate_msa = mod_params.chunk(3, dim=-1)
            else:
                shift_msa, scale_msa = mod_params.chunk(2, dim=-1)
                gate_msa = 1.0

            # Adaptive layer normalization and modulation.
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            pose_tokens_modulated = self.trunk(pose_tokens_modulated)
            # Compute the delta update for the pose encoding.
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            activated_pose = self.apply_pose_activation(pred_pose_enc)
            pred_pose_enc_list.append(activated_pose)

        return pred_pose_enc_list

    def apply_pose_activation(self, pred_pose_enc: torch.Tensor) -> torch.Tensor:
        """
        Apply activations to pose encoding, with special handling for PP variant.

        For "absT_quaR_FoV":
            - Use generic activation across T, quat, FoV via activate_pose.
        For "absT_quaR_FoV_PP":
            - Apply trans/quaternion activations as configured.
            - Apply fl_act only to FoV components (indices 7:9).
            - Keep principal point offsets (indices 9:11) linear (no activation),
              matching pose encoding in pose_enc.py where they are signed offsets.
        """
        if self.pose_encoding_type == "absT_quaR_FoV":
            T = pred_pose_enc[..., :3]
            quat = pred_pose_enc[..., 3:7]
            fl = pred_pose_enc[..., 7:]  # or fov

            T = base_pose_act(T, self.trans_act)
            quat = base_pose_act(quat, self.quat_act)
            fl = base_pose_act(fl, self.fl_act)  # or fov
            fl = fl + 0.01 # for stability

            pred_pose_enc = torch.cat([T, quat, fl], dim=-1)
            return pred_pose_enc
        elif self.pose_encoding_type == "absT_quaR_FoV_PP":
            T = pred_pose_enc[..., :3]
            quat = pred_pose_enc[..., 3:7]
            fov = pred_pose_enc[..., 7:9]
            pp_offsets = pred_pose_enc[..., 9:11]

            T = base_pose_act(T, self.trans_act)
            quat = base_pose_act(quat, self.quat_act)
            fov = base_pose_act(fov, self.fl_act)
            fov = fov + 0.01 # for stability
            
            pp_offsets = pp_offsets
            # Keep principal point offsets linear (may be negative or positive)
            pred_pose_enc = torch.cat([T, quat, fov, pp_offsets], dim=-1)
            return pred_pose_enc
        else:
            raise ValueError(f"Unsupported camera encoding type: {self.pose_encoding_type}")


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift
