#!/usr/bin/env python3
"""
Stable Depth Transformer (SDT) Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.reshape(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).reshape(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale).reshape(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").reshape(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


class WeightedFusion(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, out_channels, bias=False), nn.GELU())
            for in_dim in in_channels
        ])

        self.readout_projects = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * in_dim, in_dim), nn.GELU())
            for in_dim in in_channels
        ])

        self.layer_weights = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        assert len(features) == len(self.projections)
        projected_layer_tokens = []

        for i, layer_feature in enumerate(features):
            if isinstance(layer_feature, tuple):
                spatial_tensor, cls_token = layer_feature
                B, C, H, W = spatial_tensor.shape
                spatial_tokens = spatial_tensor.flatten(2).permute(0, 2, 1).contiguous()
                cls_token_expanded = cls_token.unsqueeze(1).expand_as(spatial_tokens)
                tokens_with_cls = torch.cat((spatial_tokens, cls_token_expanded), dim=-1)
                enhanced_tokens = self.readout_projects[i](tokens_with_cls)
                projected_tokens = self.projections[i](enhanced_tokens)
            else:
                if layer_feature.dim() == 4:
                    B, C, H, W = layer_feature.shape
                    layer_tokens = layer_feature.flatten(2).permute(0, 2, 1).contiguous()
                else:
                    layer_tokens = layer_feature
                projected_tokens = self.projections[i](layer_tokens)

            projected_layer_tokens.append(projected_tokens)

        layer_weights = F.softmax(self.layer_weights, dim=0)
        fused_tokens = torch.zeros_like(projected_layer_tokens[0])
        for i, projected_tokens in enumerate(projected_layer_tokens):
            fused_tokens = fused_tokens + layer_weights[i] * projected_tokens

        return fused_tokens


class SpatialDetailEnhancer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.activation(x + residual)
        return x


class DySampleUpsamplerWrapper(nn.Module):
    def __init__(self, feature_dim: int, scale_factor: int = 4, style: str = 'lp', groups: int = 4, dyscope: bool = False):
        super().__init__()
        self.scale_factor = scale_factor
        self.feature_dim = feature_dim

        self.dysample1 = nn.Sequential(
            DySample(feature_dim, scale=2, style=style, groups=groups, dyscope=dyscope),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True))
        self.dysample2 = nn.Sequential(
            DySample(feature_dim, scale=2, style=style, groups=groups, dyscope=dyscope),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True))

    def forward(self, features: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        x = self.dysample1(features)
        x = self.dysample2(x)
        return x


class SDTHead(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        fusion_channels: int = 256,
        n_output_channels: int = 256,
        use_cls_token: bool = False,
        **kwargs
    ):
        super().__init__()
        assert len(in_channels) == 4

        self.use_cls_token = use_cls_token
        self.fusion_channels = fusion_channels

        self.weighted_fusion = WeightedFusion(in_channels, fusion_channels)
        self.detail_enhancer = SpatialDetailEnhancer(fusion_channels)

        self.upsample_1 = DySampleUpsamplerWrapper(fusion_channels, scale_factor=4, style='lp', groups=4, dyscope=True)
        self.refinement_1 = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True))

        self.upsample_2 = DySampleUpsamplerWrapper(fusion_channels, scale_factor=4, style='lp', groups=4, dyscope=True)
        self.refinement_2 = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True))

        self.output_conv = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_output_channels, kernel_size=1))

    def forward(self, features):
        if isinstance(features[0], tuple):
            spatial_tensors = [f[0] for f in features]
            features_with_cls_token = features
        else:
            spatial_tensors = features
            features_with_cls_token = features

        B = spatial_tensors[0].shape[0]
        H_patches = spatial_tensors[0].shape[2]
        W_patches = spatial_tensors[0].shape[3]

        fused_tokens = self.weighted_fusion(features_with_cls_token)
        fused_spatial = fused_tokens.permute(0, 2, 1).contiguous().reshape(B, self.fusion_channels, H_patches, W_patches)
        enhanced_spatial = self.detail_enhancer(fused_spatial)

        x = self.upsample_1(enhanced_spatial, None)
        x = self.refinement_1(x)
        x = self.upsample_2(x, None)
        x = self.refinement_2(x)

        return self.output_conv(x)
    