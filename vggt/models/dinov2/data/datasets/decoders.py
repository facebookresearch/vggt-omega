# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from io import BytesIO
from typing import Any, Type

from PIL import Image
import numpy as np
import torch
from enum import Enum

try:
    import tifffile
except ImportError:
    print("Could not import `tifffile`, TIFFImageDataDecoder will be disabled")


class Decoder:
    def decode(self) -> Any:
        raise NotImplementedError


class DecoderType(Enum):
    ImageDataDecoder = "ImageDataDecoder"
    XChannelsDecoder = "XChannelsDecoder"
    XChannelsTIFFDecoder = "XChannelsTIFFDecoder"
    ChannelSelectDecoder = "ChannelSelectDecoder"

    def get_class(self) -> Type[Decoder]:  # noqa: C901
        if self == DecoderType.ImageDataDecoder:
            return ImageDataDecoder
        if self == DecoderType.XChannelsDecoder:
            return XChannelsDecoder
        if self == DecoderType.XChannelsTIFFDecoder:
            return XChannelsTIFFDecoder
        if self == DecoderType.ChannelSelectDecoder:
            return ChannelSelectDecoder


class ImageDataDecoder(Decoder):
    def __init__(self, image_data: bytes) -> None:
        self._image_data = image_data

    def decode(self) -> Image:
        f = BytesIO(self._image_data)
        return Image.open(f).convert(mode="RGB")


class TargetDecoder(Decoder):
    def __init__(self, target: Any):
        self._target = target

    def decode(self) -> Any:
        return self._target


class XChannelsDecoder(Decoder):
    def __init__(self, image_data: bytes) -> None:
        self._image_data = image_data

    def decode(self):
        im = np.asarray(Image.open(BytesIO(self._image_data)))
        if len(im.shape) == 2:
            im = np.reshape(im, (im.shape[0], im.shape[0], -1), order="F")
        return torch.Tensor(im).permute(2, 0, 1)


class XChannelsTIFFDecoder(Decoder):
    def __init__(self, image_data: bytes, num_channels: int = 3) -> None:
        self._image_data = image_data
        self._num_channels = num_channels

    def decode(self):
        numpy_array = tifffile.imread(BytesIO(self._image_data))
        numpy_array = np.reshape(numpy_array, (numpy_array.shape[0], -1, self._num_channels), order="F")
        return torch.Tensor(numpy_array).permute(2, 0, 1)


class ChannelSelectDecoder(Decoder):
    def __init__(self, image_data: bytes, select_channel: bool = False) -> None:
        self.select_channel = select_channel
        if select_channel:
            self._image_data = image_data[:-1]
            self._channel = image_data[-1]
        else:
            self._image_data = image_data

    def decode(self):
        im = np.asarray(Image.open(BytesIO(self._image_data)))
        if self.select_channel:
            return torch.Tensor(im).permute(2, 0, 1)[[self._channel]]
        return torch.Tensor(im).permute(2, 0, 1)
