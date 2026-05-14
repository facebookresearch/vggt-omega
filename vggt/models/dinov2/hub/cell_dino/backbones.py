# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC licence,
# found in the LICENSE_CELL_DINO_CODE file in the root directory of this source tree.

from enum import Enum
from typing import Optional, Union

import torch


class Weights(Enum):
    CELL_DINO = "CELL-DINO"


def _make_cell_dino_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    num_register_tokens: int = 0,
    interpolate_antialias: bool = False,
    interpolate_offset: float = 0.1,
    pretrained: bool = True,
    channel_adaptive: bool = False,
    weights: Union[Weights, str] = Weights.CELL_DINO,
    pretrained_url: Optional[str] = None,
    pretrained_path: Optional[str] = None,
    **kwargs,
):
    from ...models import vision_transformer as vits

    if isinstance(weights, str):
        try:
            weights = Weights[weights]
        except KeyError:
            raise AssertionError(f"Unsupported weights: {weights}")

    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
        channel_adaptive=channel_adaptive,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    if pretrained:
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location="cpu")
        else:
            pretrained_url is not None
            state_dict = torch.hub.load_state_dict_from_url(pretrained_url, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    return model


def cell_dino_hpa_vitl16(
    *,
    pretrained_url: Optional[str] = None,
    pretrained_path: Optional[str] = None,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.CELL_DINO,
    in_channels: int = 4,
    **kwargs,
):
    """
    Cell-DINO ViT-L/16 model dataset pretrained on HPA dataset.
    """
    return _make_cell_dino_model(
        arch_name="vit_large",
        patch_size=16,
        img_size=224,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        block_chunks=4,
        pretrained_url=pretrained_url,
        pretrained_path=pretrained_path,
        pretrained=pretrained,
        weights=weights,
        in_chans=in_channels,
        **kwargs,
    )


def cell_dino_hpa_vitl14(
    *,
    pretrained_url: Optional[str] = None,
    pretrained_path: Optional[str] = None,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.CELL_DINO,
    in_channels: int = 4,
    **kwargs,
):
    """
    Cell-DINO ViT-L/14 model dataset pretrained on LVD, then on HPA dataset.
    """
    return _make_cell_dino_model(
        arch_name="vit_large",
        patch_size=14,
        img_size=518,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        block_chunks=4,
        pretrained_url=pretrained_url,
        pretrained_path=pretrained_path,
        pretrained=pretrained,
        weights=weights,
        in_chans=in_channels,
        **kwargs,
    )


def cell_dino_cp_vits8(
    *,
    pretrained_url: Optional[str] = None,
    pretrained_path: Optional[str] = None,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.CELL_DINO,
    in_channels: int = 5,
    **kwargs,
):
    """
    Cell-DINO ViT-S/8 model dataset pretrained on the combined cell painting dataset.
    """
    return _make_cell_dino_model(
        arch_name="vit_small",
        patch_size=8,
        img_size=128,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        block_chunks=4,
        pretrained_url=pretrained_url,
        pretrained_path=pretrained_path,
        pretrained=pretrained,
        weights=weights,
        in_chans=in_channels,
        **kwargs,
    )


def channel_adaptive_dino_vitl16(
    *,
    pretrained_url: Optional[str] = None,
    pretrained_path: Optional[str] = None,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.CELL_DINO,
    in_channels: int = 1,
    channel_adaptive: bool = True,
    **kwargs,
):
    """
    Cell-DINO ViT-L/16 model dataset pretrained on HPA dataset.
    """
    return _make_cell_dino_model(
        arch_name="vit_large",
        patch_size=16,
        img_size=224,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        block_chunks=4,
        pretrained_url=pretrained_url,
        pretrained_path=pretrained_path,
        pretrained=pretrained,
        weights=weights,
        in_chans=in_channels,
        channel_adaptive=channel_adaptive,
        **kwargs,
    )
