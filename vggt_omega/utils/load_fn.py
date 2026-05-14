# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF


def load_and_preprocess_images_square(image_path_list, target_size=512):
    """Load images, center-pad them to square, and resize to target_size."""
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    original_coords = []
    to_tensor = TF.ToTensor()

    for image_path in image_path_list:
        image = _load_rgb_image(image_path)
        width, height = image.size

        square_size = max(width, height)
        left = (square_size - width) // 2
        top = (square_size - height) // 2
        scale = target_size / square_size

        original_coords.append(
            np.array(
                [
                    left * scale,
                    top * scale,
                    (left + width) * scale,
                    (top + height) * scale,
                    width,
                    height,
                ]
            )
        )

        square_image = Image.new("RGB", (square_size, square_size), (0, 0, 0))
        square_image.paste(image, (left, top))
        square_image = square_image.resize((target_size, target_size), Image.Resampling.BICUBIC)
        images.append(to_tensor(square_image))

    return torch.stack(images), torch.from_numpy(np.array(original_coords)).float()


def load_and_preprocess_images(image_path_list, mode="crop", target_size=512, patch_size=16):
    """Load images for VGGT-Omega inference.

    `crop` resizes each image to target_size width and center-crops tall images.
    `pad` preserves the full image by resizing the long side to target_size and
    padding the short side.
    """
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()

    for image_path in image_path_list:
        image = _load_rgb_image(image_path)
        width, height = image.size

        if mode == "pad" and width < height:
            new_height = target_size
            new_width = round(width * (new_height / height) / patch_size) * patch_size
        else:
            new_width = target_size
            new_height = round(height * (new_width / width) / patch_size) * patch_size

        new_width = max(new_width, patch_size)
        new_height = max(new_height, patch_size)

        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        image = to_tensor(image)

        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            image = image[:, start_y : start_y + target_size, :]

        if mode == "pad":
            h_padding = target_size - image.shape[1]
            w_padding = target_size - image.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                image = torch.nn.functional.pad(
                    image,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )

        shapes.add((image.shape[1], image.shape[2]))
        images.append(image)

    if len(shapes) > 1:
        warnings.warn(f"Found images with different shapes: {shapes}; padding to a common size.", stacklevel=2)
        images = _pad_images_to_common_size(images, shapes)

    return torch.stack(images)


def _load_rgb_image(image_path):
    with Image.open(image_path) as image:
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        return image.convert("RGB")


def _pad_images_to_common_size(images, shapes):
    max_height = max(shape[0] for shape in shapes)
    max_width = max(shape[1] for shape in shapes)

    padded_images = []
    for image in images:
        h_padding = max_height - image.shape[1]
        w_padding = max_width - image.shape[2]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            image = torch.nn.functional.pad(
                image,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=1.0,
            )
        padded_images.append(image)

    return padded_images
