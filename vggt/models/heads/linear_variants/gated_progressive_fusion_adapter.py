from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvUnit(nn.Module):
    def __init__(self, channels: int, use_group_norm: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm1 = nn.GroupNorm(1, channels) if use_group_norm else nn.Identity()
        self.norm2 = nn.GroupNorm(1, channels) if use_group_norm else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return out + x


class GatedFusionBlock(nn.Module):
    def __init__(self, channels: int, rcu_use_group_norm: bool = False) -> None:
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(1, channels),
            nn.Sigmoid(),
        )
        self.rcu = ResidualConvUnit(channels=channels, use_group_norm=rcu_use_group_norm)
        # Cold-start gate to prefer deep features at initialization.
        nn.init.constant_(self.gate_conv[1].weight, 0.001)
        nn.init.constant_(self.gate_conv[1].bias, -3.0)

    def forward(self, x_deep: torch.Tensor, x_shallow: torch.Tensor) -> torch.Tensor:
        gate = self.gate_conv(torch.cat([x_deep, x_shallow], dim=1))
        fused = x_deep + gate * x_shallow
        return self.rcu(fused)


class _UpsampleRepairBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        use_pointwise_conv: bool = True,
    ) -> None:
        super().__init__()
        self.dw_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            groups=channels,
            bias=True,
        )
        self.act = nn.GELU()
        if use_pointwise_conv:
            self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
            nn.init.normal_(self.pw_conv.weight, mean=0, std=1e-4)
            nn.init.zeros_(self.pw_conv.bias)
        else:
            self.pw_conv = nn.Identity()

    def forward(self, x: torch.Tensor, scale_factor: int) -> torch.Tensor:
        x_up = F.interpolate(
            x,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        )
        y = self.dw_conv(x_up)
        y = self.act(y)
        y = self.pw_conv(y)
        return x_up + y


class UpsampleToQuarter(nn.Module):
    def __init__(
        self,
        channels: int,
        two_stage_upsample: bool = False,
        use_pointwise_conv: bool = True,
    ) -> None:
        super().__init__()
        self.two_stage_upsample = two_stage_upsample
        self.repair_block_1 = _UpsampleRepairBlock(
            channels=channels,
            use_pointwise_conv=use_pointwise_conv,
        )
        self.repair_block_2 = (
            _UpsampleRepairBlock(
                channels=channels,
                use_pointwise_conv=use_pointwise_conv,
            )
            if two_stage_upsample
            else None
        )

    def forward(self, x_16: torch.Tensor) -> torch.Tensor:
        if self.two_stage_upsample:
            x = self.repair_block_1(x_16, scale_factor=2)
            return self.repair_block_2(x, scale_factor=2)
        return self.repair_block_1(x_16, scale_factor=4)


class GatedProgressiveFusionAdapter(nn.Module):
    """
    Progressive gated fusion from deep to shallow.
    - 4-branch mode: F4 -> F3 -> F2 -> F1, then optional 1/16 -> 1/4 upsample.
    - 1-branch mode: use the last-layer feature only, then optional upsample.
    """

    def __init__(
        self,
        out_channels: List[int],
        align_channels: bool = True,
        fusion_channels: int = 256,
        two_stage_upsample: bool = False,
        upsample_use_pointwise_conv: bool = True,
        rcu_use_group_norm: bool = False,
        include_upsample: bool = True,
    ) -> None:
        super().__init__()
        if len(out_channels) not in (1, 4):
            raise ValueError(
                "GatedProgressiveFusionAdapter expects 1 or 4 branches. "
                f"Got {len(out_channels)}"
            )
        self.num_branches = len(out_channels)

        self.align_channels = align_channels
        if align_channels:
            if fusion_channels <= 0:
                raise ValueError(f"fusion_channels must be > 0, got {fusion_channels}")
            channels = fusion_channels
            self.align_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(in_channels=in_ch, out_channels=channels, kernel_size=1, bias=False),
                    )
                    for in_ch in out_channels
                ]
            )
        else:
            if len(set(out_channels)) != 1:
                raise ValueError(
                    "When align_channels=False, all input channels must be equal. "
                    f"Got {out_channels}"
                )
            channels = out_channels[0]
            self.align_layers = None

        self.seed_rcu = ResidualConvUnit(
            channels=channels,
            use_group_norm=rcu_use_group_norm,
        )
        self.fuse_blocks = nn.ModuleList(
            [
                GatedFusionBlock(
                    channels=channels,
                    rcu_use_group_norm=rcu_use_group_norm,
                )
                for _ in range(self.num_branches - 1)
            ]
        )
        self.fuse_rcu = ResidualConvUnit(
            channels=channels,
            use_group_norm=rcu_use_group_norm,
        )
        if include_upsample:
            self.upsample_to_quarter = UpsampleToQuarter(
                channels=channels,
                two_stage_upsample=two_stage_upsample,
                use_pointwise_conv=upsample_use_pointwise_conv,
            )
        else:
            self.upsample_to_quarter = None

    def _maybe_align(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.align_layers is None:
            return features
        return [align(feature) for align, feature in zip(self.align_layers, features)]

    def forward(
        self,
        features: List[torch.Tensor],
        return_quarter: bool = True,
    ) -> torch.Tensor:
        if len(features) != self.num_branches:
            raise ValueError(
                "GatedProgressiveFusionAdapter expects feature maps matching the "
                f"configured branch count ({self.num_branches}), "
                f"got {len(features)}."
            )

        aligned_features = self._maybe_align(features)
        x = self.seed_rcu(aligned_features[-1])
        for block_idx, shallow_feature in enumerate(reversed(aligned_features[:-1])):
            x = self.fuse_blocks[block_idx](x, shallow_feature)
        x_16 = x
        x_16 = self.fuse_rcu(x_16)

        if return_quarter:
            if self.upsample_to_quarter is None:
                raise RuntimeError(
                    "return_quarter=True but upsample_to_quarter was not created "
                    "(include_upsample=False at init). Set fuser_output_scale=4 or "
                    "include_upsample=True to enable upsampling."
                )
            return self.upsample_to_quarter(x_16)
        return x_16

