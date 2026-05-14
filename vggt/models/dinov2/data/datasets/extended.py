# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Any, Tuple

from torchvision.datasets import VisionDataset

from .decoders import DecoderType, TargetDecoder


class ExtendedVisionDataset(VisionDataset):
    def __init__(self, *args, **kwargs) -> None:
        image_decoder_type = kwargs.pop("image_decoder_type", DecoderType.ImageDataDecoder)
        self._decoder_params = {}
        self._image_decoder_class = image_decoder_type.get_class()
        if "image_decoder_params" in kwargs:
            self._decoder_params = kwargs.pop("image_decoder_params")

        super().__init__(*args, **kwargs)  # type: ignore

    def get_image_data(self, index: int) -> bytes:
        raise NotImplementedError

    def get_target(self, index: int) -> Any:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = self.get_image_data(index)
            image = self._image_decoder_class(image_data, **self._decoder_params).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)
        target = TargetDecoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        raise NotImplementedError
