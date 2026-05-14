import argparse
import concurrent.futures
import logging
import os
from io import BytesIO
from pathlib import Path

import fsspec
import numpy as np
import yaml
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def get_fs(path: str):
    if path.startswith("s3://"):
        return fsspec.filesystem("s3", profile="dino"), path, True
    return fsspec.filesystem("file"), path, False


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def get_dataset_id(yaml_path: Path) -> str:
    return f"{yaml_path.parent.name}__{yaml_path.stem}"


def resolve_scene_list_file(config: dict, yaml_path: Path) -> Path | None:
    scene_list_value = config.get("sequence_list_file") or config.get("scene_list_file")
    if not scene_list_value:
        return None

    scene_list_path = Path(scene_list_value).expanduser()
    if not scene_list_path.is_absolute():
        scene_list_path = (yaml_path.parent / scene_list_path).resolve()
    return scene_list_path


def find_scenes(data_dir: str, fs, scene_list_file: Path | None = None) -> list[str]:
    scenes: list[str] = []
    logging.info("Searching for scenes in %s", data_dir)

    if not fs.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    if scene_list_file:
        logging.info("Loading scenes from file: %s", scene_list_file)
        if not scene_list_file.is_file():
            raise FileNotFoundError(f"Scene list file not found: {scene_list_file}")

        with scene_list_file.open("r", encoding="utf-8") as f:
            scene_names = [line.strip() for line in f if line.strip()]

        for scene_name in scene_names:
            scenes.append(os.path.join(data_dir, scene_name))
    else:
        logging.info("No scene list file provided, listing all scene directories.")
        items = fs.ls(data_dir, detail=True)
        for item in items:
            if item["type"] == "directory":
                scenes.append(item["name"])

    logging.info("Found %d scenes.", len(scenes))
    return sorted(scenes)


def scene_name_from_path(scene_path: str, data_dir: str) -> str:
    scene_path_clean = scene_path.rstrip("/")
    data_dir_clean = data_dir.rstrip("/")
    if scene_path_clean.startswith(data_dir_clean):
        return scene_path_clean[len(data_dir_clean) :].lstrip("/")
    return os.path.basename(scene_path_clean)


def check_scene_intrinsics(scene_path: str, fs, data_dir: str) -> tuple[str, str, str | None]:
    scene_name = scene_name_from_path(scene_path, data_dir)
    intrinsics_path = os.path.join(scene_path, "intrinsics.npy")

    try:
        with fs.open(intrinsics_path, "rb") as f:
            intrinsics = np.load(BytesIO(f.read()))
    except FileNotFoundError:
        return scene_name, "error", f"Missing intrinsics file: {intrinsics_path}"
    except Exception as exc:
        return scene_name, "error", f"Failed to load intrinsics: {exc}"

    intrinsics = np.asarray(intrinsics)
    negative_mask = intrinsics < 0
    if np.any(negative_mask):
        min_value = float(intrinsics[negative_mask].min())
        negative_count = int(negative_mask.sum())
        reason = f"negative_count={negative_count}, min_value={min_value:.6g}"
        return scene_name, "bad", reason

    return scene_name, "clean", None


def write_scene_list(path: Path, rows: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row}\n")


def collect_dataset_yaml_files(dataset_dirs: list[Path]) -> list[Path]:
    yaml_files: list[Path] = []
    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        yaml_files.extend(sorted(dataset_dir.glob("*.yaml")))
    return sorted(yaml_files, key=lambda path: (path.parent.name, path.name))


def check_single_dataset(
    dataset_yaml: Path,
    output_dir: Path,
    num_workers: int,
) -> dict:
    dataset_id = get_dataset_id(dataset_yaml)
    logging.info("Checking dataset yaml: %s", dataset_yaml)

    bad_scenes: dict[str, str] = {}
    clean_scenes: list[str] = []
    error_scenes: dict[str, str] = {}

    try:
        config = load_yaml(dataset_yaml)
        data_dir = config.get("UNIFIED_DIR")
        if not data_dir:
            raise ValueError("UNIFIED_DIR not found in dataset yaml")

        scene_list_file = resolve_scene_list_file(config, dataset_yaml)
        fs, data_dir, is_s3 = get_fs(data_dir)
        logging.info(
            "Dataset %s uses filesystem %s (is_s3=%s)",
            dataset_id,
            fs.protocol,
            is_s3,
        )
        scenes = find_scenes(data_dir, fs, scene_list_file)
    except Exception as exc:
        dataset_error_path = output_dir / f"{dataset_id}_dataset_error.txt"
        write_scene_list(dataset_error_path, [str(exc)])
        return {
            "dataset_id": dataset_id,
            "dataset_yaml": str(dataset_yaml),
            "num_scenes": 0,
            "num_bad": 0,
            "num_clean": 0,
            "num_error": 0,
            "dataset_error": str(exc),
            "bad_examples": [],
        }

    with tqdm(
        total=len(scenes),
        desc=f"Checking {dataset_id}",
        unit="scene",
        leave=False,
    ) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_scene = {
                executor.submit(check_scene_intrinsics, scene_path, fs, data_dir): scene_path
                for scene_path in scenes
            }
            for future in concurrent.futures.as_completed(future_to_scene):
                scene_path = future_to_scene[future]
                scene_name = scene_name_from_path(scene_path, data_dir)
                try:
                    result_scene_name, status, reason = future.result()
                except Exception as exc:
                    error_scenes[scene_name] = f"Unexpected error: {exc}"
                    pbar.set_postfix_str(f"error={scene_name}")
                    pbar.update(1)
                    continue

                if status == "bad":
                    bad_scenes[result_scene_name] = reason or "negative intrinsic found"
                    pbar.set_postfix_str(f"bad={result_scene_name}")
                elif status == "error":
                    error_scenes[result_scene_name] = reason or "unknown error"
                    pbar.set_postfix_str(f"error={result_scene_name}")
                else:
                    clean_scenes.append(result_scene_name)
                    pbar.set_postfix_str(f"ok={result_scene_name}")
                pbar.update(1)

    bad_rows = [f"{name}: {reason}" for name, reason in sorted(bad_scenes.items())]
    error_rows = [f"{name}: {reason}" for name, reason in sorted(error_scenes.items())]
    clean_rows = sorted(clean_scenes)

    write_scene_list(output_dir / f"{dataset_id}_bad.txt", bad_rows)
    write_scene_list(output_dir / f"{dataset_id}_error.txt", error_rows)
    write_scene_list(output_dir / f"{dataset_id}_clean.txt", clean_rows)

    return {
        "dataset_id": dataset_id,
        "dataset_yaml": str(dataset_yaml),
        "num_scenes": len(scenes),
        "num_bad": len(bad_scenes),
        "num_clean": len(clean_scenes),
        "num_error": len(error_scenes),
        "dataset_error": None,
        "bad_examples": [f"{dataset_id}/{name}" for name in sorted(bad_scenes)],
    }


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    cw_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description=(
            "Check all datasets under cw/datasets_real and cw/datasets_synt, "
            "and mark sequences whose intrinsics.npy contains negative values as bad examples."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-yaml-files",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Optional exact dataset yaml paths to run. "
            "When provided, this takes priority over --dataset-dirs and --dataset-names."
        ),
    )
    parser.add_argument(
        "--dataset-dirs",
        type=Path,
        nargs="+",
        default=[cw_dir / "datasets_real", cw_dir / "datasets_synt"],
        help="Dataset yaml directories to scan.",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="*",
        default=None,
        help=(
            "Optional dataset yaml stems to run, for example `omniworld scannet`. "
            "If not provided, all yaml files under dataset dirs are checked."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of worker threads used to check scenes within each dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "checks",
        help="Directory used to save per-dataset outputs and summary files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_yaml_files:
        dataset_yaml_files = [path.expanduser().resolve() for path in args.dataset_yaml_files]
    else:
        dataset_dirs = [path.expanduser().resolve() for path in args.dataset_dirs]
        dataset_yaml_files = collect_dataset_yaml_files(dataset_dirs)

        if args.dataset_names:
            requested_names = set(args.dataset_names)
            dataset_yaml_files = [
                path for path in dataset_yaml_files if path.stem in requested_names
            ]

    if not dataset_yaml_files:
        logging.error("No dataset yaml files matched the requested filters.")
        return 1

    summary_lines: list[str] = []
    all_bad_examples: list[str] = []
    dataset_error_count = 0

    for dataset_yaml in dataset_yaml_files:
        result = check_single_dataset(
            dataset_yaml=dataset_yaml,
            output_dir=output_dir,
            num_workers=args.num_workers,
        )

        if result["dataset_error"]:
            dataset_error_count += 1
            summary_lines.append(
                f"{result['dataset_id']}: DATASET_ERROR: {result['dataset_error']}"
            )
            logging.error(
                "Dataset %s failed before scene checking: %s",
                result["dataset_id"],
                result["dataset_error"],
            )
            continue

        summary_lines.append(
            (
                f"{result['dataset_id']}: total={result['num_scenes']}, "
                f"bad={result['num_bad']}, clean={result['num_clean']}, "
                f"error={result['num_error']}"
            )
        )
        all_bad_examples.extend(result["bad_examples"])
        logging.info(
            "Finished %s: total=%d, bad=%d, clean=%d, error=%d",
            result["dataset_id"],
            result["num_scenes"],
            result["num_bad"],
            result["num_clean"],
            result["num_error"],
        )

    write_scene_list(output_dir / "summary.txt", summary_lines)
    write_scene_list(output_dir / "all_bad_examples.txt", sorted(all_bad_examples))

    logging.info("Checked %d dataset yaml files.", len(dataset_yaml_files))
    logging.info("Datasets with setup errors: %d", dataset_error_count)
    logging.info("Total bad examples found: %d", len(all_bad_examples))
    logging.info("Summary saved to %s", output_dir / "summary.txt")
    logging.info("All bad examples saved to %s", output_dir / "all_bad_examples.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
