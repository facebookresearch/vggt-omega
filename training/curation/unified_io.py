# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np


def read_sequence_list(path: str | Path) -> list[str]:
    names = []
    seen = set()
    for line_number, raw_line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if any(character.isspace() for character in line):
            raise ValueError(f"{path}:{line_number}: expected one sequence name")
        validate_relative_name(line, "Sequence name")
        if line in seen:
            raise ValueError(f"{path}:{line_number}: duplicate sequence {line!r}")
        seen.add(line)
        names.append(line)
    if not names:
        raise ValueError(f"Sequence list is empty: {path}")
    return names


def list_sequence_names(data_root: str | Path, sequence_list: str | Path | None = None) -> list[str]:
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Unified data root is not a directory: {root}")
    if sequence_list is not None:
        return read_sequence_list(sequence_list)
    names = sorted(path.name for path in root.iterdir() if path.is_dir())
    if not names:
        raise ValueError(f"Unified data root contains no sequence directories: {root}")
    return names


def load_sequence(
    data_root: str | Path,
    sequence_name: str,
    mask_path: str | None = None,
) -> dict:
    validate_relative_name(sequence_name, "Sequence name")
    if mask_path:
        validate_relative_name(mask_path, "Mask directory")

    sequence_dir = Path(data_root) / sequence_name
    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"Sequence directory does not exist: {sequence_dir}")
    names_path = sequence_dir / "image_names.json"
    try:
        image_names = json.loads(names_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {names_path}: {error}") from error
    _validate_image_names(image_names, names_path)

    frame_count = len(image_names)
    cam_from_worlds = _load_array(sequence_dir / "cam_from_worlds.npy", (frame_count, 3, 4))
    intrinsics = _load_array(sequence_dir / "intrinsics.npy", (frame_count, 3, 3))
    rotations = cam_from_worlds[:, :, :3]
    identity_errors = np.max(np.abs(rotations @ np.transpose(rotations, (0, 2, 1)) - np.eye(3)), axis=(1, 2))
    if np.any(identity_errors > 5e-3) or np.any(np.linalg.det(rotations) <= 0):
        raise ValueError(f"Camera rotations are not valid rotation matrices: {sequence_dir / 'cam_from_worlds.npy'}")
    if np.any(intrinsics[:, 0, 0] <= 0) or np.any(intrinsics[:, 1, 1] <= 0):
        raise ValueError(f"Focal lengths must be positive: {sequence_dir / 'intrinsics.npy'}")

    images = []
    depths = []
    for image_name in image_names:
        image_file = sequence_dir / "images" / image_name
        depth_file = sequence_dir / "depths" / f"{Path(image_name).stem}.exr"
        image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image is missing or unreadable: {image_file}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Depth is missing or unreadable: {depth_file}")
        if depth.ndim != 2:
            raise ValueError(f"Depth must be H x W, got {depth.shape}: {depth_file}")
        if image.shape[:2] != depth.shape:
            raise ValueError(f"Image/depth shape mismatch for {image_name}: {image.shape[:2]} vs {depth.shape}")
        depth = depth.astype(np.float32, copy=False)

        if mask_path:
            depth = _apply_mask(depth, sequence_dir / mask_path / f"{Path(image_name).stem}.png")
        images.append(image)
        depths.append(depth)

    return {
        "sequence_name": sequence_name,
        "image_names": image_names,
        "images": images,
        "depths": depths,
        "cam_from_worlds": cam_from_worlds,
        "intrinsics": intrinsics,
    }


def validate_relative_name(name: str, description: str) -> None:
    path = Path(name)
    if not name or name == "." or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{description} must be a safe relative path: {name!r}")


def _validate_image_names(image_names: object, path: Path) -> None:
    if not isinstance(image_names, list) or not image_names:
        raise ValueError(f"{path} must contain a non-empty JSON list")
    seen = set()
    for image_name in image_names:
        if not isinstance(image_name, str) or Path(image_name).name != image_name:
            raise ValueError(f"Invalid image name in {path}: {image_name!r}")
        if Path(image_name).suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise ValueError(f"Unsupported image extension in {path}: {image_name!r}")
        if image_name in seen:
            raise ValueError(f"Duplicate image name in {path}: {image_name!r}")
        seen.add(image_name)


def _load_array(path: Path, expected_shape: tuple[int, ...]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required array does not exist: {path}")
    array = np.load(path, allow_pickle=False)
    if array.shape != expected_shape:
        raise ValueError(f"Expected {expected_shape} at {path}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"Array must contain only finite numeric values: {path}")
    return array.astype(np.float64, copy=False)


def _apply_mask(depth: np.ndarray, path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Configured mask is missing or unreadable: {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape != depth.shape:
        mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    invalid = mask == 0
    return np.where(invalid, 0.0, depth)
