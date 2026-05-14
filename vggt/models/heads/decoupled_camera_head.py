# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from vggt.models.layers.block import SelfAttentionBlock
from vggt.models.layers.ffn_layers import Mlp
from vggt.models.heads.head_act import base_pose_act


class DecoupledCameraHead(nn.Module):
    """
    DecoupledCameraHead predicts camera extrinsic and intrinsic parameters from
    separate tokens.

    It applies a series of transformer blocks to two dedicated camera tokens,
    then predicts extrinsic and intrinsic parameters using separate MLPs.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 1e-5,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",
        use_qk_norm: bool = False,
    ):
        super().__init__()

        self.extri_dim = 7  # 3 for translation, 4 for quaternion
        self.intri_dim = 2  # 2 for focal length / fov
        self.target_dim = self.extri_dim + self.intri_dim

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                SelfAttentionBlock(dim=dim_in, num_heads=num_heads, ffn_ratio=mlp_ratio, init_values=init_values, use_qk_norm=use_qk_norm)
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in, eps=1e-5)
        self.trunk_norm = nn.LayerNorm(dim_in, eps=1e-5)

        self.extri_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.extri_dim, drop=0)
        self.intri_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.intri_dim, drop=0)

    def forward(self, aggregated_tokens_list: list) -> list:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.

        Returns:
            list: A list containing the predicted camera encoding (post-activation).
        """
        # Use tokens from the last block for camera prediction.
        tokens = aggregated_tokens_list[-1]

        # Extract the first two tokens for extrinsic and intrinsic parameters.
        # pose_tokens = tokens[:, :2, :]
        pose_tokens = tokens[:, :, :2, :]
        
        B, S, N, C = pose_tokens.shape
        
        pose_tokens = pose_tokens.reshape(B, S * N, C)
        pose_tokens = self.token_norm(pose_tokens)

        # Process tokens through the transformer trunk.
        pose_tokens_modulated = self.trunk(pose_tokens)
        pose_tokens_modulated = self.trunk_norm(pose_tokens_modulated)
        
        pose_tokens_modulated = pose_tokens_modulated.reshape(B, S, N, C)

        # Separate tokens for extrinsic and intrinsic prediction.
        extri_token = pose_tokens_modulated[:, :, 0]
        intri_token = pose_tokens_modulated[:, :, 1]

        # Predict extrinsic and intrinsic parameters.
        pred_extri_enc = self.extri_branch(extri_token)
        pred_intri_enc = self.intri_branch(intri_token)

        # Concatenate to form the full pose encoding.
        pred_pose_enc = torch.cat([pred_extri_enc, pred_intri_enc], dim=-1)

        # Apply activation functions.
        activated_pose = self.apply_pose_activation(pred_pose_enc)

        return [activated_pose]

    def apply_pose_activation(self, pred_pose_enc: torch.Tensor) -> torch.Tensor:
        """
        Apply activations to the concatenated pose encoding.

        The encoding is split into translation, quaternion, and focal length components,
        and respective activation functions are applied.
        """
        T = pred_pose_enc[..., :3]
        quat = pred_pose_enc[..., 3:7]
        fl = pred_pose_enc[..., 7:]  # focal length / fov

        T = base_pose_act(T, self.trans_act)
        quat = base_pose_act(quat, self.quat_act)
        fl = base_pose_act(fl, self.fl_act)

        pred_pose_enc = torch.cat([T, quat, fl], dim=-1)
        return pred_pose_enc
