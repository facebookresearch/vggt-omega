from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(ROOT))

Dataset = importlib.import_module("datasets").Dataset
metric_module = importlib.import_module("metrics")
crop_depth = metric_module.crop_depth
depth_metrics = metric_module.depth_metrics
pose_errors = metric_module.pose_errors
pose_metrics = metric_module.pose_metrics
resize_depth = metric_module.resize_depth
Predictor = importlib.import_module("model").Predictor


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("eth3d", "sintel"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-scenes", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    frame_path = ROOT / "frames" / f"{args.dataset}.json"
    frame_map = json.loads(frame_path.read_text())
    if args.max_scenes is not None:
        frame_map = dict(list(frame_map.items())[: args.max_scenes])
    dataset = Dataset(args.data_root)
    predictor = Predictor(args.checkpoint, args.device)
    rotations = []
    translations = []
    per_scene = {}
    frame_names = {}
    for scene, indices in frame_map.items():
        batch = dataset.load(scene, indices)
        predicted_depth, predicted_extrinsics, crop = predictor.predict(
            batch.image_paths
        )
        target_depth = crop_depth(batch.depth, crop)
        predicted_depth = resize_depth(predicted_depth, target_depth.shape[-2:])
        depth = depth_metrics(predicted_depth, target_depth)
        rotation, translation = pose_errors(predicted_extrinsics, batch.extrinsics)
        rotations.append(rotation)
        translations.append(translation)
        frame_names[scene] = batch.frame_names
        per_scene[scene] = {
            **depth,
            "rotation_mean": float(rotation.mean()),
            "translation_mean": float(translation.mean()),
        }
        print(
            f"{scene}: delta125={depth['delta125'] * 100:.4f} AbsRel={depth['AbsRel']:.6f}"
        )
    rotation = np.concatenate(rotations)
    translation = np.concatenate(translations)
    camera = pose_metrics(rotation, translation)
    summary = {
        "AUC@3": camera["AUC@3"],
        "AUC@30": camera["AUC@30"],
        "delta125": float(np.mean([row["delta125"] for row in per_scene.values()]))
        * 100.0,
        "AbsRel": float(np.mean([row["AbsRel"] for row in per_scene.values()])),
    }
    output = {
        "checkpoint": Path(args.checkpoint).name,
        "dataset": args.dataset,
        "elapsed_seconds": round(time.time() - started, 1),
        "environment": {
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "frame_map_sha256": _sha256(frame_path),
        "frame_names": frame_names,
        "frames": sum(map(len, frame_map.values())),
        "metrics": summary,
        "per_scene": per_scene,
        "pose_pairs": int(rotation.size),
        "scenes": len(frame_map),
    }
    _write_json(Path(args.output).expanduser().resolve(), output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
