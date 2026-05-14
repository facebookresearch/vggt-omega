# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the licence
# found in the LICENSE_XRAY_DINO_MODEL file in the root directory of this source tree.

from typing import Union

from ..backbones import Weights, _make_dinov2_model


def xray_dino_vitl16(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.XRAY_DINO, **kwargs):
    """
    XRay-DINO ViT-L/16 model (optionally) pretrained on the XRay-DINO dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_large",
        patch_size=16,
        img_size=512,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        block_chunks=4,
        pretrained=pretrained,
        weights=weights,
        hash="ad31c2b0",
        check_hash=True,
        **kwargs,
    )
