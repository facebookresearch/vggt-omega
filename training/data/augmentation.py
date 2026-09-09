# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import v2 as transforms_v2


class JpegCompressionFloat:
    """
    Wrapper around transforms_v2.JPEG that handles float tensor input.

    transforms_v2.JPEG requires uint8 input, but our pipeline uses float32 [0,1].
    This wrapper converts float32 -> uint8, applies JPEG, then converts back.
    """

    def __init__(self, quality=(50, 95)):
        self.jpeg = transforms_v2.JPEG(quality=quality)

    def __call__(self, img):
        # Convert float [0,1] to uint8 [0,255]
        img_uint8 = (img * 255).clamp(0, 255).to(torch.uint8)
        # Apply JPEG compression
        img_uint8 = self.jpeg(img_uint8)
        # Convert back to float [0,1]
        return img_uint8.to(img.dtype) / 255.0


class LowResBlur:
    """Simulate low-resolution blur by downsampling and upsampling."""

    def __init__(self, resize_ratio_range=(0.25, 1.0)):
        self.resize_ratio_range = resize_ratio_range
        self.downsample_modes = ["bicubic", "area", "bilinear"]
        self.upsample_modes = ["bicubic", "bilinear", "nearest"]

    def _resize(self, img, size, mode):
        *leading_dims, channels, height, width = img.shape
        flat_img = img.reshape(-1, channels, height, width)

        resize_kwargs = {
            "size": size,
            "mode": mode,
        }
        if mode in {"bilinear", "bicubic"}:
            resize_kwargs["align_corners"] = False
            resize_kwargs["antialias"] = True

        flat_img = F.interpolate(flat_img, **resize_kwargs)
        return flat_img.reshape(*leading_dims, channels, size[0], size[1])

    def __call__(self, img):
        # img: [..., C, H, W]
        if img.ndim < 3:
            raise ValueError(
                f"Expected image tensor with shape [..., C, H, W], got {tuple(img.shape)}"
            )

        _, H, W = img.shape[-3:]
        ratio = torch.empty(1).uniform_(*self.resize_ratio_range).item()
        new_H = max(1, int(H * ratio))
        new_W = max(1, int(W * ratio))

        downsample_idx = torch.randint(0, len(self.downsample_modes), (1,)).item()
        downsample_mode = self.downsample_modes[downsample_idx]
        img = self._resize(img, (new_H, new_W), downsample_mode)

        upsample_idx = torch.randint(0, len(self.upsample_modes), (1,)).item()
        upsample_mode = self.upsample_modes[upsample_idx]
        img = self._resize(img, (H, W), upsample_mode)
        return img


def get_image_augmentation(
    color_jitter: Optional[Dict[str, float]] = None,
    gray_scale: bool = True,
    gau_blur: bool = False,
    jpeg_compression: float = -1,
    low_res_blur: float = -1,
    low_res_blur_min_ratio: float = 0.25,
) -> Optional[transforms.Compose]:
    """Create a composition of image augmentations.

    Args:
        color_jitter: Dictionary containing color jitter parameters:
            - brightness: float (default: 0.5)
            - contrast: float (default: 0.5)
            - saturation: float (default: 0.5)
            - hue: float (default: 0.1)
            - p: probability of applying (default: 0.9)
            If None, uses default values
        gray_scale: Whether to apply random grayscale (default: True)
        gau_blur: Whether to apply gaussian blur (default: False)
        jpeg_compression: Probability of applying JPEG compression. If < 0, disabled (default: -1)
        low_res_blur: Probability of applying low-res blur. If < 0, disabled (default: -1)
        low_res_blur_min_ratio: Minimum resize ratio sampled by low-res blur (default: 0.25)

    Returns:
        A Compose object of transforms or None if no transforms are added
    """
    transform_list = []
    default_jitter = {
        "brightness": 0.5,
        "contrast": 0.5,
        "saturation": 0.5,
        "hue": 0.1,
        "p": 0.9,
    }

    # Handle color jitter
    if color_jitter is not None:
        # Merge with defaults for missing keys
        effective_jitter = {**default_jitter, **color_jitter}
    else:
        effective_jitter = default_jitter

    transform_list.append(
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=effective_jitter["brightness"],
                    contrast=effective_jitter["contrast"],
                    saturation=effective_jitter["saturation"],
                    hue=effective_jitter["hue"],
                )
            ],
            p=effective_jitter["p"],
        )
    )

    if gray_scale:
        transform_list.append(transforms.RandomGrayscale(p=0.05))

    if gau_blur:
        transform_list.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.05
            )
        )

    if jpeg_compression > 0:
        transform_list.append(
            transforms.RandomApply(
                [JpegCompressionFloat(quality=(50, 99))], p=jpeg_compression
            )
        )

    if low_res_blur > 0:
        transform_list.append(
            transforms.RandomApply(
                [LowResBlur(resize_ratio_range=(low_res_blur_min_ratio, 1.0))],
                p=low_res_blur,
            )
        )

    return transforms.Compose(transform_list) if transform_list else None
