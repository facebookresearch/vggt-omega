# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .head_act import activate_head
from vggt.models.layers.ffn_layers import Mlp

class LinearFeatHead(nn.Module):
    """
    Linear Head for feature descriptor learning.
    This head computes descriptors and an InfoNCE loss for metric learning.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        patch_size: int = 16,
        output_dim: int = 33,
        activation: str = "norm",
        conf_activation: str = "expp0",
        intermediate_layer_idx: List[int] = None,
        enhance_mlp_ratio: float = -1,
        mlp_ratio: float = 2.0,
        proj_type = "mlp_self",
        eps = 1e-5,
        temperature: float = 0.5,
        alpha: float = 10,
        predict_tokens_dim: int = -1,
        **kwargs,
    ) -> None:
        super(LinearFeatHead, self).__init__()
        
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.activation = activation
        self.conf_activation = conf_activation
        self.intermediate_layer_idx = intermediate_layer_idx
        self.temperature = temperature
        self.alpha = alpha
        self.eps = eps
        self.predict_tokens_dim = predict_tokens_dim

        self.norm = nn.LayerNorm(dim_in, eps=eps)
        
        if enhance_mlp_ratio > 0:
            self.enhance_mlp = nn.Sequential(
                Mlp(dim_in, int(dim_in * enhance_mlp_ratio)),
                nn.LayerNorm(dim_in, eps=eps),
            )
        else:
            self.enhance_mlp = None
            
        if intermediate_layer_idx is not None:
            proj_input_dim = dim_in * len(intermediate_layer_idx)
        else:
            proj_input_dim = dim_in
            
        self.token_proj = None
        
        if self.predict_tokens_dim > 0:
            # self.token_proj = nn.Linear(proj_input_dim, self.predict_tokens_dim)
            self.token_proj = Mlp(proj_input_dim, int(proj_input_dim * mlp_ratio), self.predict_tokens_dim)
        else:
            if proj_type == "linear":
                self.proj = nn.Linear(proj_input_dim, output_dim * self.patch_size ** 2)
            elif proj_type == "mlp":
                self.proj = nn.Sequential(
                    Mlp(proj_input_dim, int(proj_input_dim * mlp_ratio)),
                    nn.LayerNorm(proj_input_dim, eps=eps),
                    nn.Linear(proj_input_dim, output_dim * self.patch_size ** 2),
                )
            elif proj_type == "mlp_self":
                self.proj = Mlp(proj_input_dim, int(proj_input_dim * mlp_ratio), output_dim * self.patch_size ** 2)
            else:
                raise ValueError(f"Unknown projection type: {proj_type}")

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        batch: dict,
        patch_start_idx: int,
    ) -> Union[dict, torch.Tensor]:
        """
        Forward pass through the LinearFeatHead head.
        """
        images = batch["images"]
        B, S, _, H, W = images.shape

        if self.intermediate_layer_idx is not None:
            tokens = []
            for layer_idx in self.intermediate_layer_idx:
                token = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
                token = self.norm(token)
                if self.enhance_mlp is not None:
                    token = self.enhance_mlp(token)
                tokens.append(token)
            tokens = torch.cat(tokens, dim=-1)
            _, _, P, C = tokens.shape
            tokens = tokens.view(B * S, P, C)
        else:
            tokens = aggregated_tokens_list[-1][:, :, patch_start_idx:]
            _, _, P, C = tokens.shape
            tokens = tokens.view(B * S, P, C)
            tokens = self.norm(tokens)
            if self.enhance_mlp is not None:
                tokens = self.enhance_mlp(tokens)

        if self.predict_tokens_dim > 0:
            tokens = self.token_proj(tokens)
            tokens = F.normalize(tokens, dim=-1)
            
            tokens = tokens.view(B, S, P, tokens.shape[-1])
            output_dict = {"descriptor": tokens}
            return output_dict
        
        feat = self.proj(tokens)

        feat = feat.transpose(-1, -2).contiguous()        
        feat = feat.view(B * S, -1, H // self.patch_size, W // self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # [B*S, output_dim, H, W]

        descriptor, conf = activate_head(feat, activation=self.activation, conf_activation=self.conf_activation)

        descriptor = descriptor.view(B, S, *descriptor.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])

        output_dict = {
            "descriptor": descriptor,
            "descriptor_conf": conf,
        }
        return output_dict
