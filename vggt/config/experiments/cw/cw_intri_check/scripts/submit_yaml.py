import argparse
import os
import subprocess
from pathlib import Path

import yaml


def submit_job(yaml_file: Path, num_workers: int, output_dir: Path | None) -> None:
    yaml_file = yaml_file.expanduser().resolve()
    if not yaml_file.exists():
        print(f"Error: {yaml_file} does not exist.")
        return

    try:
        with yaml_file.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"Error reading {yaml_file}: {exc}")
        return

    unified_dir = config.get("UNIFIED_DIR")
    if not unified_dir:
        print(f"Skipping {yaml_file}: UNIFIED_DIR not found.")
        return

    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    script_path = current_dir / "submit_job.sh"
    check_script_path = parent_dir / "check_negative_intrinsics.py"
    logs_dir = current_dir / "outputs"
    logs_dir.mkdir(exist_ok=True)

    dataset_id = f"{yaml_file.parent.name}_{yaml_file.stem}"
    job_name = f"intri_{dataset_id}"
    output_log = logs_dir / f"{job_name}_%j.out"
    error_log = logs_dir / f"{job_name}_%j.err"

    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--output={output_log}",
        f"--error={error_log}",
        f"--export=ALL,CHECK_SCRIPT={check_script_path}",
        str(script_path),
        f"--dataset-yaml-files={yaml_file}",
        f"--num_workers={num_workers}",
    ]

    if output_dir is not None:
        cmd.append(f"--output-dir={output_dir.expanduser().resolve()}")

    print(f"Submitting intrinsic check for {dataset_id}...")
    subprocess.run(cmd, check=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit one intrinsic-check job from a dataset yaml config."
    )
    parser.add_argument("yaml_file", type=Path, help="Path to the dataset yaml file.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=96,
        help="Number of worker threads used inside the cluster job.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for checker result files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    submit_job(args.yaml_file, num_workers=args.num_workers, output_dir=args.output_dir)
