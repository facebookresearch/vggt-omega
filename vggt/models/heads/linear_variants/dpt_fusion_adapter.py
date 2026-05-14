from typing import List, Tuple

import torch
import torch.nn as nn

class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(
        self,
        features: int,
        activation: nn.Module,
        bn: bool,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )
        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )
        self.norm1 = None
        self.norm2 = None
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features: int,
        activation: nn.Module,
        deconv: bool = False,
        bn: bool = False,
        expand: bool = False,
        align_corners: bool = True,
        size: Tuple[int, int] = None,
        has_residual: bool = True,
        groups: int = 1,
        use_identity_resconfunit1: bool = False,
    ) -> None:
        super().__init__()
        assert use_identity_resconfunit1 == False, "use_identity_resconfunit1 must be False"
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        self.resConfUnit1 = None
        out_features = features // 2 if self.expand else features
        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=self.groups,
        )
        if has_residual:
            if use_identity_resconfunit1:
                self.resConfUnit1 = None
            else:
                self.resConfUnit1 = ResidualConvUnit(
                    features,
                    activation,
                    bn,
                    groups=self.groups,
                )
        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(
            features,
            activation,
            bn,
            groups=self.groups,
        )
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(
        self,
        *xs: torch.Tensor,
        size: Tuple[int, int] = None,
    ) -> torch.Tensor:
        output = xs[0]
        if self.has_residual:
            if self.resConfUnit1 is None:
                res = xs[1]
            else:
                res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        
        output = custom_interpolate(
            output,
            **modifier,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        output = self.out_conv(output)
        return output


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    if size is None:
        size = (
            int(x.shape[-2] * scale_factor),
            int(x.shape[-1] * scale_factor),
        )

    if tuple(x.shape[-2:]) == tuple(size):
        return x

    int_max = 1610612736
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > int_max:
        chunks = torch.chunk(x, chunks=(input_elements // int_max) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(
                chunk,
                size=size,
                mode=mode,
                align_corners=align_corners,
            )
            for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()

    return nn.functional.interpolate(
        x,
        size=size,
        mode=mode,
        align_corners=align_corners,
    )


def _make_fusion_block(
    features: int,
    size: Tuple[int, int] = None,
    has_residual: bool = True,
    groups: int = 1,
    align_corners: bool = True,
    use_identity_resconfunit1: bool = False,
) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=align_corners,
        size=size,
        has_residual=has_residual,
        groups=groups,
        use_identity_resconfunit1=use_identity_resconfunit1,
    )


def _make_scratch(
    in_shape: List[int],
    out_shape: int,
    groups: int = 1,
    expand: bool = False,
) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3],
            out_shape4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )

    return scratch


class _DPTResizeLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        resize_scale: float,
        use_interpolate_conv2d_upsample: bool = False,
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        if resize_scale not in (0.5, 1.0, 2.0, 4.0):
            raise ValueError(
                "Unsupported resize_scale for DPT resize layer. "
                f"Expected one of [0.5, 1.0, 2.0, 4.0], got {resize_scale}."
            )

        if resize_scale == 1.0:
            self.layer = nn.Identity()
        elif resize_scale < 1.0:
            downsample_stride = int(1.0 / resize_scale)
            self.layer = nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=downsample_stride,
                padding=1,
            )
        else:
            upsample_scale = int(resize_scale)
            if use_interpolate_conv2d_upsample:
                self.layer = nn.Sequential(
                    nn.Upsample(
                        scale_factor=upsample_scale,
                        mode="bilinear",
                        align_corners=align_corners,
                    ),
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
            else:
                self.layer = nn.ConvTranspose2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=upsample_scale,
                    stride=upsample_scale,
                    padding=0,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class DPTFusionAdapter(nn.Module):
    """
    DPT-equivalent resize + scratch fusion module (without output_conv1/output_conv2).
    """

    def __init__(
        self,
        out_channels: List[int],
        fusion_features: int = 256,
        use_interpolate_conv2d_upsample: bool = False,
        align_corners: bool = True,
        use_identity_resconfunit1: bool = False,
    ) -> None:
        super().__init__()
        if len(out_channels) != 4:
            raise ValueError(
                "DPT-equivalent fusion requires exactly 4 branches. "
                f"Got {len(out_channels)}"
            )

        self.use_interpolate_conv2d_upsample = use_interpolate_conv2d_upsample
        self.align_corners = align_corners
        self.use_identity_resconfunit1 = use_identity_resconfunit1
        resize_scales = [4.0, 2.0, 1.0, 0.5]
        self.resize_layers = nn.ModuleList(
            [
                _DPTResizeLayer(
                    channels=channels,
                    resize_scale=resize_scale,
                    use_interpolate_conv2d_upsample=self.use_interpolate_conv2d_upsample,
                    align_corners=self.align_corners,
                )
                for channels, resize_scale in zip(out_channels, resize_scales)
            ]
        )

        self.scratch = _make_scratch(
            in_shape=out_channels,
            out_shape=fusion_features,
            expand=False,
        )
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(
            fusion_features,
            align_corners=self.align_corners,
            use_identity_resconfunit1=self.use_identity_resconfunit1,
        )
        self.scratch.refinenet2 = _make_fusion_block(
            fusion_features,
            align_corners=self.align_corners,
            use_identity_resconfunit1=self.use_identity_resconfunit1,
        )
        self.scratch.refinenet3 = _make_fusion_block(
            fusion_features,
            align_corners=self.align_corners,
            use_identity_resconfunit1=self.use_identity_resconfunit1,
        )
        self.scratch.refinenet4 = _make_fusion_block(
            fusion_features,
            has_residual=False,
            align_corners=self.align_corners,
            use_identity_resconfunit1=self.use_identity_resconfunit1,
        )

        
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError(
                "DPT-equivalent fusion expects 4 feature maps, "
                f"got {len(features)}."
            )

        resized_features = [
            resize_layer(feature_map)
            for resize_layer, feature_map in zip(self.resize_layers, features)
        ]

        layer_1, layer_2, layer_3, layer_4 = resized_features
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        # Keep output at layer_1 scale (1/4 for patch_size=16).
        out = self.scratch.refinenet1(out, layer_1_rn, size=layer_1_rn.shape[2:])
        del layer_1_rn, layer_1
        return out

