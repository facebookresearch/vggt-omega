# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC licence,
# found in the LICENSE_CELL_DINO_CODE file in the root directory of this source tree.

import logging
import torchvision
from torchvision import transforms

from .transforms import (
    RandomContrastProteinChannel,
    RandomRemoveChannelExceptProtein,
    RandomBrightness,
    RandomContrast,
    Div255,
    SelfNormalizeNoDiv,
)

logger = logging.getLogger("dinov2")


class CellAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        additional_transforms_list = [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            RandomBrightness(),
            RandomContrast(),
            SelfNormalizeNoDiv(),
        ]

        first_transforms_list = [
            Div255(),
            RandomRemoveChannelExceptProtein(),
            RandomContrastProteinChannel(),
        ]

        global_transforms_list = first_transforms_list.copy()
        global_transforms_list.append(
            torchvision.transforms.RandomResizedCrop(size=global_crops_size, scale=global_crops_scale)
        )
        global_transforms_list = global_transforms_list + additional_transforms_list

        local_transforms_list = first_transforms_list
        local_transforms_list.append(
            torchvision.transforms.RandomResizedCrop(size=local_crops_size, scale=local_crops_scale)
        )
        local_transforms_list = local_transforms_list + additional_transforms_list

        self.global_transform = transforms.Compose(global_transforms_list)
        self.local_transform = transforms.Compose(local_transforms_list)

    def __call__(self, image):
        output = {}

        global_crop1 = self.global_transform(image)
        global_crop2 = self.global_transform(image)

        output["global_crops"] = [global_crop1, global_crop2]

        local_crops = []
        for _ in range(self.local_crops_number):
            local_crops.append(self.local_transform(image))

        output["local_crops"] = local_crops
        output["global_crops_teacher"] = [global_crop1, global_crop2]
        output["offsets"] = ()

        return output
