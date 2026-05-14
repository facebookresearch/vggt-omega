# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC licence,
# found in the LICENSE_CELL_DINO_CODE file in the root directory of this source tree.

import torch
from torchvision import transforms
import numpy as np
from enum import Enum


class NormalizationType(Enum):
    SELF_NORM_AUG_DECODER = "self_norm_aug_decoder"
    SELF_NORM_CENTER_CROP = "self_norm_center_crop"


class Div255(torch.nn.Module):
    def forward(self, x):
        x = x / 255
        return x


class SelfNormalizeNoDiv(torch.nn.Module):
    def forward(self, x):
        m = x.mean((-2, -1), keepdim=True)
        s = x.std((-2, -1), unbiased=False, keepdim=True)
        x -= m
        x /= s + 1e-7
        return x


class SelfNormalize(torch.nn.Module):
    def forward(self, x):
        x = x / 255
        m = x.mean((-2, -1), keepdim=True)
        s = x.std((-2, -1), unbiased=False, keepdim=True)
        x -= m
        x /= s + 1e-7
        return x


class RandomContrastProteinChannel(torch.nn.Module):
    """
    Random constrast rescaling of the protein channel only.
    RescaleProtein function in Dino4cell codebase.
    """

    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, img):
        if img.max() == 0:
            return img
        if len(img) == 1:
            return img
        if np.random.rand() <= self.p:
            random_factor = (np.random.rand() * 2) / img.max()  # scaling
            img[1] = img[1] * random_factor
            return img
        else:
            return img


class RandomRemoveChannelExceptProtein(torch.nn.Module):
    """
    dropping a channel at random except the channel 1, corresponding to proteins in HPA datasets.
    """

    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, img):
        img_size = np.array(img).shape
        if img_size[0] < 4:
            return img
        if np.random.rand() <= self.p:
            channel_to_blacken = np.random.choice(np.array([0, 2, 3]))
            img[channel_to_blacken] = torch.zeros(1, *img.shape[1:])
            return img
        else:
            return img


class RandomRemoveChannel(torch.nn.Module):
    """
    dropping a channel at random
    """

    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, img):
        img_size = np.array(img).shape
        num_channels = img_size[0]
        if num_channels < 4:
            return img
        if np.random.rand() <= self.p:
            channel_to_blacken = np.random.choice(np.array(list(range(num_channels))))
            img[channel_to_blacken] = torch.zeros(1, *img.shape[1:])
            return img
        else:
            return img


class RandomContrast(torch.nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, img):
        if img.max() == 0:
            return img
        n_channels = img.shape[0]
        for ind in range(n_channels):
            factor = max(np.random.normal(1, self.p), 0.5)
            img[ind] = transforms.functional.adjust_contrast(img[ind][None, ...], factor)
        return img


class RandomBrightness(torch.nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, img):
        if img.max() == 0:
            return img
        n_channels = img.shape[0]
        for ind in range(n_channels):
            factor = max(np.random.normal(1, self.p), 0.5)
            img[ind] = transforms.functional.adjust_brightness(img[ind], factor)
        return img


def make_classification_eval_cell_transform(
    *,
    resize_size: int = 0,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 384,
    normalization_type: Enum = NormalizationType.SELF_NORM_CENTER_CROP,
) -> transforms.Compose:

    from .transforms import (
        Div255,
        SelfNormalizeNoDiv,
    )

    transforms_list = [Div255()]
    if resize_size > 0:
        transforms_list.append(transforms.Resize(resize_size, interpolation=interpolation))

    if normalization_type == NormalizationType.SELF_NORM_AUG_DECODER:
        transforms_list.extend(
            [
                transforms.RandomCrop(size=crop_size, pad_if_needed=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
    elif normalization_type == NormalizationType.SELF_NORM_CENTER_CROP:
        transforms_list.append(transforms.CenterCrop(size=crop_size))
    else:
        raise ValueError("f{normalization_type}: unknown NormalizationType")
    transforms_list.append(SelfNormalizeNoDiv())

    return transforms.Compose(transforms_list)
