#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Template for exporting loader samples to a static visual-review site.

The reusable code in this file deliberately knows nothing about a particular
dataset or Hydra configuration. Copy it into a work directory, implement
``iter_samples`` using the repository's real data loader, and keep the rest of
the exporter unchanged unless the review needs additional views or metadata.

Run the built-in smoke test from the repository root:

    python training/dataset_preparation/visualize_dataset_template.py \
        --demo --output /tmp/vggt_omega_review_demo
    python -m http.server 8000 --directory /tmp/vggt_omega_review_demo

The generated page loads its manifest with ``fetch``, so serve the directory
over HTTP instead of opening ``index.html`` with a ``file://`` URL.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image


DEFAULT_MODEL_VIEWER_SCRIPT = (
    "https://ajax.googleapis.com/ajax/libs/model-viewer/4.1.0/model-viewer.min.js"
)


@dataclass
class ReviewSample:
    """One unbatched multi-view sample ready for visual export.

    ``images`` may be N x H x W x 3 or N x 3 x H x W. Floating-point images
    must be display-ready in [0, 1] or [0, 255]. ``depths`` are optical-axis
    z-depths. ``extrinsics`` are OpenCV camera-from-world matrices. When
    ``world_points`` is omitted, this template unprojects depth using the same
    pixel-centre convention as the public training loader. An optional
    ``point_masks`` array preserves the usable point support supplied by a
    later loader stage.
    """

    sample_id: str
    images: Any
    depths: Any
    extrinsics: Any
    intrinsics: Any
    world_points: Any | None = None
    point_masks: Any | None = None
    frame_labels: Sequence[str] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    data_stage: str = "loader output"


def review_sample_from_loader(
    sample: Mapping[str, Any],
    *,
    sample_id: str,
    frame_labels: Sequence[str] | None = None,
    data_stage: str = "loader output",
    metadata: Mapping[str, Any] | None = None,
) -> ReviewSample:
    """Adapt the fields returned by the public VGGT-Omega data pipeline."""

    required = ("images", "depths", "extrinsics", "intrinsics")
    missing = [key for key in required if key not in sample]
    if missing:
        raise KeyError(f"Loader sample is missing required fields: {missing}")

    merged_metadata = dict(metadata or {})
    for key in ("dataset", "seq_name", "is_synthetic", "depth_train_mask"):
        if key in sample:
            merged_metadata.setdefault(key, _jsonable(sample[key]))

    return ReviewSample(
        sample_id=sample_id,
        images=sample["images"],
        depths=sample["depths"],
        extrinsics=sample["extrinsics"],
        intrinsics=sample["intrinsics"],
        world_points=sample.get("world_points"),
        point_masks=sample.get("point_masks"),
        frame_labels=frame_labels,
        metadata=merged_metadata,
        data_stage=data_stage,
    )


def iter_samples(args: argparse.Namespace) -> Iterable[ReviewSample]:
    """Yield real loader samples after adapting this function for a dataset.

    Keep samples unbatched. Prefer ``review_sample_from_loader`` when the
    loader already returns VGGT-Omega's standard sample dictionary. Record the
    resolved configuration and stable frame identifiers in ``metadata`` when
    they are available.
    """

    raise NotImplementedError(
        "Implement iter_samples() in a working copy of this template, or run "
        "with --demo to exercise the exporter without a dataset."
    )


def export_review_site(
    samples: Iterable[ReviewSample],
    output_dir: Path | str,
    *,
    title: str = "VGGT-Omega dataset review",
    max_points: int = 200_000,
    show_cameras: bool = False,
    model_viewer_script: str = DEFAULT_MODEL_VIEWER_SCRIPT,
    repo_root: Path | str | None = None,
) -> Path:
    """Export samples and return the generated manifest path.

    The output directory must be absent or empty. This avoids mixing assets
    from two reviews. Large or resumable jobs should preserve this manifest
    contract while adding dataset-specific sharding outside this template.
    """

    if max_points <= 0:
        raise ValueError("max_points must be positive")
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose a new directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir()

    glb_exporter = _load_glb_exporter(repo_root)
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for sample_index, sample in enumerate(samples):
        if not isinstance(sample.sample_id, str) or not sample.sample_id.strip():
            raise ValueError("Every sample_id must be a nonempty string")
        if sample.sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id: {sample.sample_id!r}")
        seen_ids.add(sample.sample_id)
        record = _export_one_sample(
            sample,
            sample_index,
            samples_dir,
            max_points=max_points,
            show_cameras=show_cameras,
            glb_exporter=glb_exporter,
        )
        records.append(record)

    if not records:
        raise ValueError("iter_samples() produced no samples")

    generated_at = datetime.now(timezone.utc).isoformat()
    run_identity = json.dumps(
        {"title": title, "samples": records},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "title": title,
        "run_id": hashlib.sha256(run_identity).hexdigest()[:16],
        "asset_revision": hashlib.sha256(generated_at.encode("utf-8")).hexdigest()[:12],
        "generated_at": generated_at,
        "sample_count": len(records),
        "review_labels": [
            "unreviewed",
            "good",
            "possible_view_discontinuity",
            "other_issue",
            "uncertain",
        ],
        "samples": records,
    }
    manifest_path = output_dir / "manifest.json"
    temporary_manifest = output_dir / "manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)

    (output_dir / "index.html").write_text(
        _site_html(title, model_viewer_script), encoding="utf-8"
    )
    print(f"Exported {len(records)} samples to {output_dir}")
    print(f"Serve with: python -m http.server 8000 --directory {output_dir}")
    return manifest_path


def _export_one_sample(
    sample: ReviewSample,
    sample_index: int,
    samples_dir: Path,
    *,
    max_points: int,
    show_cameras: bool,
    glb_exporter: Any,
) -> dict[str, Any]:
    arrays = _normalize_sample_arrays(sample)
    images = arrays["images"]
    depths = arrays["depths"]
    extrinsics = arrays["extrinsics"]
    world_points = arrays["world_points"]
    point_masks = arrays["point_masks"]

    sample_slug = _slug(sample.sample_id)
    relative_sample_dir = Path("samples") / f"{sample_index:05d}_{sample_slug}"
    sample_dir = samples_dir / f"{sample_index:05d}_{sample_slug}"
    image_dir = sample_dir / "rgb"
    depth_frame_dir = sample_dir / "depth_frame"
    depth_shared_dir = sample_dir / "depth_shared"
    for directory in (image_dir, depth_frame_dir, depth_shared_dir):
        directory.mkdir(parents=True, exist_ok=False)

    frame_labels = list(sample.frame_labels) if sample.frame_labels is not None else []
    if not frame_labels:
        frame_labels = [f"view_{index:03d}" for index in range(len(images))]
    if len(frame_labels) != len(images):
        raise ValueError(
            f"{sample.sample_id}: frame_labels has {len(frame_labels)} entries, "
            f"but the sample has {len(images)} frames"
        )
    frame_labels = [str(label) for label in frame_labels]

    shared_range = _depth_range(depths)
    frame_records = []
    for frame_index, (label, image, depth) in enumerate(
        zip(frame_labels, images, depths, strict=True)
    ):
        filename = f"{frame_index:03d}.png"
        Image.fromarray(image).save(image_dir / filename)

        frame_range = _depth_range(depth[None])
        Image.fromarray(_colorize_depth(depth, frame_range)).save(
            depth_frame_dir / filename
        )
        Image.fromarray(_colorize_depth(depth, shared_range)).save(
            depth_shared_dir / filename
        )

        valid = np.isfinite(depth) & (depth > 0)
        rendered = (
            valid
            & point_masks[frame_index]
            & np.isfinite(world_points[frame_index]).all(axis=-1)
        )
        frame_records.append(
            {
                "label": label,
                "rgb": (relative_sample_dir / "rgb" / filename).as_posix(),
                "depth_frame": (
                    relative_sample_dir / "depth_frame" / filename
                ).as_posix(),
                "depth_shared": (
                    relative_sample_dir / "depth_shared" / filename
                ).as_posix(),
                "valid_depth_fraction": float(valid.mean()),
                "rendered_point_fraction": float(rendered.mean()),
                "depth_frame_range": list(frame_range) if frame_range else None,
            }
        )

    valid_geometry = (
        np.isfinite(depths)
        & (depths > 0)
        & point_masks
        & np.isfinite(world_points).all(axis=-1)
    )
    glb_path: str | None = None
    if np.any(valid_geometry):
        predictions = {
            "world_points_from_depth": world_points,
            "depth_conf": valid_geometry.astype(np.float32),
            "images": images.astype(np.float32) / 255.0,
            "extrinsic": extrinsics,
            "depth": depths[..., None],
        }
        scene = glb_exporter(
            predictions,
            conf_thres=2.0,
            show_cam=show_cameras,
            max_points=max_points,
            filter_depth_edges=False,
        )
        target_glb = sample_dir / "scene.glb"
        scene.export(file_obj=str(target_glb))
        glb_path = (relative_sample_dir / "scene.glb").as_posix()

    metadata = _jsonable(dict(sample.metadata))
    if not isinstance(metadata, dict):
        raise TypeError(f"{sample.sample_id}: metadata must serialize to a JSON object")
    metadata.update(
        {
            "frame_count": len(images),
            "image_shape": list(images.shape[1:3]),
            "valid_depth_fraction": float((np.isfinite(depths) & (depths > 0)).mean()),
            "rendered_point_fraction": float(valid_geometry.mean()),
            "shared_depth_range": list(shared_range) if shared_range else None,
            "camera_markers_in_glb": show_cameras,
        }
    )
    return {
        "sample_id": sample.sample_id,
        "data_stage": sample.data_stage,
        "frame_labels": frame_labels,
        "assets": {"glb": glb_path},
        "frames": frame_records,
        "metadata": metadata,
    }


def _normalize_sample_arrays(sample: ReviewSample) -> dict[str, np.ndarray]:
    images = _normalize_images(sample.images, sample.sample_id)
    depths = _to_numpy(sample.depths).astype(np.float32, copy=False)
    if depths.ndim == 4 and depths.shape[-1] == 1:
        depths = depths[..., 0]
    if depths.ndim != 3:
        raise ValueError(
            f"{sample.sample_id}: depths must be N x H x W, got {depths.shape}"
        )

    extrinsics = _to_numpy(sample.extrinsics).astype(np.float32, copy=False)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(
            f"{sample.sample_id}: extrinsics must be N x 3 x 4 or N x 4 x 4, "
            f"got {extrinsics.shape}"
        )
    extrinsics = extrinsics[:, :3, :4]

    intrinsics = _to_numpy(sample.intrinsics).astype(np.float32, copy=False)
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(
            f"{sample.sample_id}: intrinsics must be N x 3 x 3, got {intrinsics.shape}"
        )

    expected_frames = len(images)
    if not (len(depths) == len(extrinsics) == len(intrinsics) == expected_frames):
        raise ValueError(
            f"{sample.sample_id}: inconsistent frame counts for images, depths, "
            "extrinsics, and intrinsics"
        )
    if images.shape[1:3] != depths.shape[1:3]:
        raise ValueError(
            f"{sample.sample_id}: RGB shape {images.shape[1:3]} does not match "
            f"depth shape {depths.shape[1:3]}"
        )
    if not np.isfinite(extrinsics).all() or not np.isfinite(intrinsics).all():
        raise ValueError(f"{sample.sample_id}: camera arrays contain non-finite values")
    if np.any(intrinsics[:, 0, 0] <= 0) or np.any(intrinsics[:, 1, 1] <= 0):
        raise ValueError(f"{sample.sample_id}: camera focal lengths must be positive")

    if sample.world_points is None:
        world_points = _unproject_depths(depths, extrinsics, intrinsics)
    else:
        world_points = _to_numpy(sample.world_points).astype(np.float32, copy=False)
        if world_points.shape != (*depths.shape, 3):
            raise ValueError(
                f"{sample.sample_id}: world_points must have shape "
                f"{(*depths.shape, 3)}, got {world_points.shape}"
            )

    if sample.point_masks is None:
        point_masks = np.isfinite(depths) & (depths > 0)
    else:
        point_masks = _to_numpy(sample.point_masks)
        if point_masks.ndim == 4 and point_masks.shape[-1] == 1:
            point_masks = point_masks[..., 0]
        if point_masks.shape != depths.shape:
            raise ValueError(
                f"{sample.sample_id}: point_masks must have shape {depths.shape}, "
                f"got {point_masks.shape}"
            )
        point_masks = point_masks.astype(bool, copy=False)

    return {
        "images": images,
        "depths": depths,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "world_points": world_points,
        "point_masks": point_masks,
    }


def _normalize_images(images: Any, sample_id: str) -> np.ndarray:
    images = _to_numpy(images)
    if images.ndim != 4:
        raise ValueError(
            f"{sample_id}: images must be N x H x W x C or N x C x H x W, "
            f"got {images.shape}"
        )
    if not len(images):
        raise ValueError(f"{sample_id}: images contains no frames")
    if images.shape[-1] in (1, 3, 4):
        pass
    elif images.shape[1] in (1, 3, 4):
        images = np.transpose(images, (0, 2, 3, 1))
    else:
        raise ValueError(f"{sample_id}: cannot identify the image channel dimension")

    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)
    elif images.shape[-1] == 4:
        images = images[..., :3]

    if not np.isfinite(images).all():
        raise ValueError(f"{sample_id}: images contain non-finite values")
    image_min = float(images.min())
    image_max = float(images.max())
    if np.issubdtype(images.dtype, np.floating) and image_min >= 0 and image_max <= 1:
        images = images * 255.0
    elif image_min < 0 or image_max > 255:
        raise ValueError(
            f"{sample_id}: images are not display-ready; observed range "
            f"[{image_min}, {image_max}]. Undo model normalization in the adapter."
        )
    return np.rint(images).clip(0, 255).astype(np.uint8)


def _unproject_depths(
    depths: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    """Unproject z-depth using the public loader's OpenCV pixel convention."""

    world_points = []
    for depth, world_to_camera, intrinsic in zip(
        depths, extrinsics, intrinsics, strict=True
    ):
        height, width = depth.shape
        u, v = np.meshgrid(
            np.arange(width, dtype=np.float32) + 0.5,
            np.arange(height, dtype=np.float32) + 0.5,
        )
        camera_points = np.stack(
            (
                (u - intrinsic[0, 2]) * depth / intrinsic[0, 0],
                (v - intrinsic[1, 2]) * depth / intrinsic[1, 1],
                depth,
            ),
            axis=-1,
        )
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :4] = world_to_camera
        camera_to_world = np.linalg.inv(transform)
        points = camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
        world_points.append(points.astype(np.float32))
    return np.stack(world_points)


def _depth_range(depths: np.ndarray) -> tuple[float, float] | None:
    valid = depths[np.isfinite(depths) & (depths > 0)]
    if not len(valid):
        return None
    low, high = np.percentile(valid, [2.0, 98.0])
    if high <= low:
        high = low + max(abs(float(low)) * 1e-6, 1e-6)
    return float(low), float(high)


def _colorize_depth(
    depth: np.ndarray, value_range: tuple[float, float] | None
) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if value_range is None:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    low, high = value_range
    normalized = np.zeros_like(depth, dtype=np.float32)
    normalized[valid] = np.clip((depth[valid] - low) / (high - low), 0, 1)
    colored_bgr = cv2.applyColorMap(
        np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    colored = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    colored[~valid] = 0
    return colored


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if (
        isinstance(value, np.ndarray)
        or hasattr(value, "detach")
        or hasattr(value, "numpy")
    ):
        array = _to_numpy(value)
        if array.size <= 64:
            return _jsonable(array.tolist())
        return {"shape": list(array.shape), "dtype": str(array.dtype)}
    return str(value)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return slug[:80] or "sample"


def _load_glb_exporter(repo_root: Path | str | None) -> Any:
    candidates = []
    if repo_root is not None:
        candidates.append(Path(repo_root).expanduser().resolve())
    candidates.append(Path.cwd().resolve())
    candidates.extend(Path(__file__).resolve().parents)

    for candidate in candidates:
        if (candidate / "visual_util.py").is_file() and (
            candidate / "training" / "README.md"
        ).is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            from visual_util import predictions_to_glb

            return predictions_to_glb
    raise RuntimeError(
        "Could not locate the VGGT-Omega repository root containing visual_util.py. "
        "Run from the repository root or pass --repo-root."
    )


def _demo_sample() -> ReviewSample:
    frame_count, height, width = 3, 120, 160
    u = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    v = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    images, depths, extrinsics, intrinsics = [], [], [], []

    for frame_index in range(frame_count):
        red = np.broadcast_to(u, (height, width))
        green = np.broadcast_to(v, (height, width))
        blue = np.full((height, width), frame_index / (frame_count - 1))
        images.append(np.stack((red, green, blue), axis=-1))

        depth = 2.0 + 0.25 * np.sin(2 * np.pi * u) + 0.1 * frame_index
        depth = np.broadcast_to(depth, (height, width)).copy()
        depth[:8, :8] = 0
        depths.append(depth)

        world_to_camera = np.eye(4, dtype=np.float32)
        world_to_camera[0, 3] = 0.12 * (1 - frame_index)
        extrinsics.append(world_to_camera[:3])
        intrinsics.append(
            np.array(
                [[140.0, 0.0, width / 2], [0.0, 140.0, height / 2], [0, 0, 1]],
                dtype=np.float32,
            )
        )

    return ReviewSample(
        sample_id="synthetic_demo",
        images=np.stack(images),
        depths=np.stack(depths),
        extrinsics=np.stack(extrinsics),
        intrinsics=np.stack(intrinsics),
        frame_labels=["left", "centre", "right"],
        metadata={"purpose": "exporter smoke test", "units": "arbitrary"},
        data_stage="synthetic demo",
    )


def _site_html(title: str, model_viewer_script: str) -> str:
    safe_title = html.escape(title)
    safe_script = html.escape(model_viewer_script, quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <script type="module" src="{safe_script}"></script>
  <style>
    :root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #11151b; color: #e8edf2; }}
    header {{ display: flex; gap: .75rem; align-items: center; padding: .8rem 1rem; border-bottom: 1px solid #303944; }}
    header h1 {{ font-size: 1rem; margin: 0 auto 0 0; }}
    #sample-picker {{ max-width: min(34vw, 420px); }}
    input, select, textarea, button {{ color: inherit; background: #1c232c; border: 1px solid #46515f; border-radius: .35rem; padding: .45rem .6rem; }}
    button {{ cursor: pointer; }}
    article {{ max-width: 1500px; margin: auto; padding: 1rem; }}
    .summary {{ display: flex; align-items: baseline; gap: .75rem; flex-wrap: wrap; }}
    .summary h2 {{ margin: 0; font-size: 1.1rem; }}
    .badge {{ color: #a9c7e8; font-size: .85rem; }}
    model-viewer {{ display: block; width: 100%; height: min(58vh, 620px); min-height: 360px; margin: .8rem 0; background: #080a0d; border-radius: .5rem; }}
    #viewer-empty {{ display: none; place-items: center; min-height: 240px; margin: .8rem 0; background: #080a0d; color: #aab3bd; }}
    .review {{ display: grid; grid-template-columns: minmax(180px, 280px) 1fr; gap: .6rem; margin-bottom: 1rem; }}
    textarea {{ min-height: 2.5rem; resize: vertical; }}
    #frames {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: .8rem; }}
    .frame {{ border: 1px solid #303944; border-radius: .5rem; padding: .6rem; background: #171c23; }}
    .frame h3 {{ margin: 0 0 .5rem; font-size: .9rem; }}
    .frame-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .3rem; }}
    figure {{ margin: 0; min-width: 0; }}
    .frame img {{ width: 100%; display: block; background: black; }}
    figcaption {{ color: #aab3bd; font-size: .72rem; margin-top: .2rem; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; color: #bdc7d1; }}
    #error {{ color: #ff8d8d; white-space: pre-wrap; }}
    @media (max-width: 800px) {{
      header {{ flex-wrap: wrap; }}
      #sample-picker {{ max-width: 100%; }}
      .review, .frame-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{safe_title}</h1>
    <button id="previous">Previous</button>
    <select id="sample-picker"></select>
    <button id="next">Next</button>
    <span id="count"></span>
    <button id="export-reviews">Export reviews</button>
  </header>
  <main>
    <article>
      <div id="error"></div>
      <div class="summary"><h2 id="sample-title">Loading…</h2><span id="stage" class="badge"></span></div>
      <model-viewer id="viewer" camera-controls interaction-prompt="none"></model-viewer>
      <div id="viewer-empty">This sample has no valid depth geometry to display.</div>
      <div class="review">
        <select id="review-label"></select>
        <textarea id="review-notes" placeholder="Optional review notes"></textarea>
      </div>
      <div id="frames"></div>
      <details><summary>Metadata</summary><pre id="metadata"></pre></details>
    </article>
  </main>
  <script>
    const get = id => document.getElementById(id);
    const ids = ["previous", "sample-picker", "next", "count", "export-reviews",
      "sample-title", "stage", "viewer", "viewer-empty", "review-label",
      "review-notes", "frames", "metadata", "error"];
    const el = Object.fromEntries(ids.map(id => [id, get(id)]));
    let manifest, selectedId, reviews = {{}};
    const reviewFor = id => reviews[id] || {{label: "unreviewed", notes: ""}};
    function assetUrl(path) {{
      const separator = path.includes("?") ? "&" : "?";
      return `${{path}}${{separator}}v=${{encodeURIComponent(manifest.asset_revision)}}`;
    }}
    const saveReviews = () => localStorage.setItem(
      `vggt-omega-review:${{manifest.run_id}}`, JSON.stringify(reviews));
    const rangeText = range => range ?
      `${{range[0].toPrecision(4)}}–${{range[1].toPrecision(4)}}` : "no valid depth";
    function frameCard(frame, sharedRange) {{
      const card = document.createElement("section");
      card.className = "frame";
      const heading = document.createElement("h3");
      heading.textContent = `${{frame.label}} · ${{(100 * frame.valid_depth_fraction).toFixed(1)}}% depth · ${{(100 * frame.rendered_point_fraction).toFixed(1)}}% points`;
      card.appendChild(heading);
      const grid = document.createElement("div");
      grid.className = "frame-grid";
      const views = [
        [frame.rgb, "RGB"],
        [frame.depth_frame, `Depth · per-frame linear ${{rangeText(frame.depth_frame_range)}}`],
        [frame.depth_shared, `Depth · shared linear ${{rangeText(sharedRange)}}`],
      ];
      for (const [src, caption] of views) {{
        const figure = document.createElement("figure");
        const image = document.createElement("img");
        image.loading = "lazy";
        image.src = assetUrl(src);
        image.alt = `${{frame.label}} ${{caption}}`;
        const label = document.createElement("figcaption");
        label.textContent = caption;
        figure.append(image, label);
        grid.appendChild(figure);
      }}
      card.appendChild(grid);
      return card;
    }}
    function selectSample(id) {{
      const sample = manifest.samples.find(item => item.sample_id === id);
      if (!sample) return;
      selectedId = id;
      el["sample-picker"].value = id;
      const sampleIndex = manifest.samples.indexOf(sample);
      el.count.textContent = `${{sampleIndex + 1}} / ${{manifest.sample_count}}`;
      el["sample-title"].textContent = sample.sample_id;
      el.stage.textContent = sample.data_stage;
      el.metadata.textContent = JSON.stringify(sample.metadata, null, 2);
      const review = reviewFor(id);
      el["review-label"].value = review.label;
      el["review-notes"].value = review.notes;
      if (sample.assets.glb) {{
        el.viewer.style.display = "block";
        el["viewer-empty"].style.display = "none";
        el.viewer.addEventListener("load", () => {{
          el.viewer.cameraOrbit = "0deg 75deg 100%";
          if (el.viewer.jumpCameraToGoal) el.viewer.jumpCameraToGoal();
        }}, {{once: true}});
        el.viewer.src = assetUrl(sample.assets.glb);
      }} else {{
        el.viewer.removeAttribute("src");
        el.viewer.style.display = "none";
        el["viewer-empty"].style.display = "grid";
      }}
      el.frames.replaceChildren(...sample.frames.map(
        frame => frameCard(frame, sample.metadata.shared_depth_range)
      ));
    }}
    function updateReview() {{
      if (!selectedId) return;
      reviews[selectedId] = {{
        label: el["review-label"].value,
        notes: el["review-notes"].value,
      }};
      saveReviews();
    }}
    el["sample-picker"].addEventListener("change", event => selectSample(event.target.value));
    el.previous.addEventListener("click", () => {{
      const index = manifest.samples.findIndex(sample => sample.sample_id === selectedId);
      selectSample(manifest.samples[Math.max(0, index - 1)].sample_id);
    }});
    el.next.addEventListener("click", () => {{
      const index = manifest.samples.findIndex(sample => sample.sample_id === selectedId);
      selectSample(manifest.samples[Math.min(manifest.samples.length - 1, index + 1)].sample_id);
    }});
    el["review-label"].addEventListener("change", updateReview);
    el["review-notes"].addEventListener("input", updateReview);
    el["export-reviews"].addEventListener("click", () => {{
      const payload = {{run_id: manifest.run_id, reviews}};
      const link = document.createElement("a");
      link.href = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], {{type: "application/json"}}));
      link.download = `reviews-${{manifest.run_id}}.json`;
      link.click();
      URL.revokeObjectURL(link.href);
    }});
    fetch("manifest.json", {{cache: "no-store"}}).then(response => {{
      if (!response.ok) throw new Error(`manifest request failed: ${{response.status}}`);
      return response.json();
    }}).then(value => {{
      manifest = value;
      document.title = manifest.title;
      const saved = localStorage.getItem(`vggt-omega-review:${{manifest.run_id}}`);
      if (saved) {{
        try {{ reviews = JSON.parse(saved); }} catch (error) {{ console.warn(error); }}
      }}
      for (const sample of manifest.samples) {{
        const option = document.createElement("option");
        option.value = sample.sample_id;
        option.textContent = sample.sample_id;
        el["sample-picker"].appendChild(option);
      }}
      for (const label of manifest.review_labels) {{
        const option = document.createElement("option");
        option.value = label;
        option.textContent = label;
        el["review-label"].appendChild(option);
      }}
      if (manifest.samples.length) selectSample(manifest.samples[0].sample_id);
    }}).catch(error => {{
      el.error.textContent = `Could not load the review: ${{error}}\nServe this directory over HTTP instead of opening index.html directly.`;
      el["sample-title"].textContent = "Review unavailable";
    }});
  </script>
</body>
</html>
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unified-dataset-dir", type=Path)
    parser.add_argument("--dataset-config")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--title", default="VGGT-Omega dataset review")
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument("--show-cameras", action="store_true")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument(
        "--model-viewer-script",
        default=DEFAULT_MODEL_VIEWER_SCRIPT,
        help="Pinned URL or review-site-relative path to model-viewer.min.js",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Export one synthetic sample instead of calling iter_samples()",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_points <= 0:
        raise ValueError("--max-points must be positive")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    samples = [_demo_sample()] if args.demo else iter_samples(args)
    export_review_site(
        samples,
        args.output,
        title=args.title,
        max_points=args.max_points,
        show_cameras=args.show_cameras,
        model_viewer_script=args.model_viewer_script,
        repo_root=args.repo_root,
    )


if __name__ == "__main__":
    main()
