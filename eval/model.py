from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from metrics import pad_extrinsics
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def _crop_box(width: int, height: int) -> tuple[float, float, float, float]:
    aspect = height / width
    if aspect < 0.5:
        crop_width = min(width, max(1, round(height / 0.5)))
        left = max((width - crop_width) // 2, 0)
        return left / width, 0.0, (left + crop_width) / width, 1.0
    if aspect > 2.0:
        crop_height = min(height, max(1, round(width * 2.0)))
        top = max((height - crop_height) // 2, 0)
        return 0.0, top / height, 1.0, (top + crop_height) / height
    return 0.0, 0.0, 1.0, 1.0


def _source_crop(image_paths: list[str]) -> tuple[float, float, float, float]:
    sizes = []
    for path in image_paths:
        with Image.open(path) as image:
            sizes.append(image.size)
    crops = [_crop_box(width, height) for width, height in sizes]
    spread = max(
        abs(value - crops[0][axis]) for crop in crops for axis, value in enumerate(crop)
    )
    if spread > 0.004:
        raise ValueError(f"inconsistent preprocessing crops: {sorted(set(sizes))}")
    return crops[0]


class Predictor:
    def __init__(self, checkpoint: str, device: str = "cuda"):
        self.device = device
        state = torch.load(
            Path(checkpoint).expanduser().resolve(),
            map_location="cpu",
            weights_only=True,
        )
        if isinstance(state, dict):
            state = state.get("model", state.get("state_dict", state))
        self.model = VGGTOmega()
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.eval().to(device)

    def predict(
        self, image_paths: list[str]
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
        crop = _source_crop(image_paths)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            images = load_and_preprocess_images(
                image_paths,
                mode="max_size",
                image_resolution=512,
                patch_size=16,
            )
        if any("padding" in str(item.message) for item in caught):
            raise ValueError("input frames require padding")
        images = images.to(self.device)
        with torch.inference_mode():
            prediction = self.model(images)
        extrinsics, _ = encoding_to_camera(
            prediction["pose_enc"], prediction["images"].shape[-2:]
        )
        depth = prediction["depth"].detach().float().cpu().clone().numpy()[0]
        if depth.ndim == 4:
            depth = depth[..., 0]
        poses = extrinsics.detach().float().cpu().clone().numpy()[0]
        return depth.astype(np.float32), pad_extrinsics(poses), crop
