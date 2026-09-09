# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def rotate_90_degrees(
    image,
    depth_map,
    extri_opencv,
    intri_opencv,
    clockwise=True,
):
    """Rotate image-aligned data and camera parameters by 90 degrees."""
    image_height, image_width = image.shape[:2]
    rotated_image, rotated_depth_map = rotate_image_and_depth_rot90(
        image,
        depth_map,
        clockwise,
    )
    new_intri_opencv = adjust_intrinsic_matrix_rot90(
        intri_opencv,
        image_width,
        image_height,
        clockwise,
    )
    new_extri_opencv = adjust_extrinsic_matrix_rot90(
        extri_opencv,
        clockwise,
    )
    return (
        rotated_image,
        rotated_depth_map,
        new_extri_opencv,
        new_intri_opencv,
    )


def _rotate_array_rot90(array, clockwise):
    axes = (1, 0)
    rotated = np.transpose(array, axes + tuple(range(2, array.ndim)))
    return np.copy(np.flip(rotated, axis=1 if clockwise else 0))


def rotate_image_and_depth_rot90(image, depth_map, clockwise):
    rotated_image = _rotate_array_rot90(image, clockwise)
    rotated_depth_map = (
        _rotate_array_rot90(depth_map, clockwise) if depth_map is not None else None
    )
    return rotated_image, rotated_depth_map


def adjust_extrinsic_matrix_rot90(extri_opencv, clockwise):
    rotation = np.array(
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        if clockwise
        else [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        dtype=extri_opencv.dtype,
    )
    new_rotation = rotation @ extri_opencv[:, :3]
    new_translation = rotation @ extri_opencv[:, 3]
    return np.hstack((new_rotation, new_translation.reshape(-1, 1)))


def adjust_intrinsic_matrix_rot90(
    intri_opencv,
    image_width,
    image_height,
    clockwise,
):
    fx = intri_opencv[0, 0]
    fy = intri_opencv[1, 1]
    cx = intri_opencv[0, 2]
    cy = intri_opencv[1, 2]

    new_intri_opencv = np.eye(3, dtype=intri_opencv.dtype)
    new_intri_opencv[0, 0] = fy
    new_intri_opencv[1, 1] = fx
    if clockwise:
        new_intri_opencv[0, 2] = image_height - cy
        new_intri_opencv[1, 2] = cx
    else:
        new_intri_opencv[0, 2] = cy
        new_intri_opencv[1, 2] = image_width - cx
    return new_intri_opencv
