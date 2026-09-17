from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Batch:
    image_paths: list[str]
    depth: np.ndarray
    extrinsics: np.ndarray
    frame_names: list[str]


def _read_depth(path: Path) -> np.ndarray:
    compressed = path.with_suffix(".npz")
    if compressed.is_file():
        with np.load(compressed) as archive:
            return archive["depth"].astype(np.float32)
    array = path.with_suffix(".npy")
    if array.is_file():
        return np.load(array).astype(np.float32)
    raise FileNotFoundError(compressed)


def _resize_depth(depth: np.ndarray, width: int = 518) -> np.ndarray:
    source_height, source_width = depth.shape
    height = round(source_height * width / source_width / 14) * 14
    return cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)


class Dataset:
    def __init__(self, root: str):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

    def load(self, scene: str, indices: list[int]) -> Batch:
        root = self.root / scene
        names = [
            line.strip() for line in (root / "frames.txt").read_text().splitlines()
        ]
        with np.load(root / "cameras.npz") as cameras:
            extrinsics = cameras["extrinsics"]
        if len(names) != len(extrinsics):
            raise ValueError(f"camera count mismatch in {scene}")
        selected_names = [names[index] for index in indices]
        image_paths = [str(root / "images" / name) for name in selected_names]
        depths = [
            _resize_depth(_read_depth(root / "depths" / Path(name).stem))
            for name in selected_names
        ]
        for path in image_paths:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        return Batch(
            image_paths=image_paths,
            depth=np.stack(depths),
            extrinsics=extrinsics[indices],
            frame_names=selected_names,
        )
