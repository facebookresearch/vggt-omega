# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from pathlib import Path

from . import CURATION_SCHEMA_VERSION
from .compute_metrics import compute_features, read_metrics_jsonl, write_jsonl
from .ensemble import feature_matrix, load_bundle, predict_two_false
from .unified_io import list_sequence_names, load_sequence


def curate_records(bundle: dict, metrics: dict[str, dict[str, float]], errors: dict[str, str]) -> list[dict]:
    names = sorted(metrics)
    records = []
    if names:
        predictions = predict_two_false(bundle["models"], feature_matrix(metrics, names))
        for index, sequence in enumerate(names):
            base_predictions = {
                name: bool(value)
                for name, value in zip(bundle["model_names"], predictions["base_predictions"][index])
            }
            records.append(
                {
                    "schema_version": CURATION_SCHEMA_VERSION,
                    "sequence": sequence,
                    "status": "pass" if predictions["accepted"][index] else "reject",
                    "false_votes": int(predictions["false_votes"][index]),
                    "positive_score": float(predictions["positive_score"][index]),
                    "base_predictions": base_predictions,
                }
            )
    for sequence in sorted(errors):
        records.append(
            {
                "schema_version": CURATION_SCHEMA_VERSION,
                "sequence": sequence,
                "status": "error",
                "error": errors[sequence],
            }
        )
    return sorted(records, key=lambda record: record["sequence"])


def write_allowlist(records: list[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w") as handle:
        for record in records:
            if record["status"] == "pass":
                handle.write(record["sequence"] + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def _compute_from_data(args: argparse.Namespace, feature_config: dict) -> tuple[dict, dict]:
    metrics = {}
    errors = {}
    for sequence in list_sequence_names(args.data_root, args.sequence_list):
        try:
            scene = load_sequence(
                args.data_root,
                sequence,
                mask_path=feature_config["mask_path"],
            )
            metrics[sequence] = compute_features(
                scene,
                seed=feature_config["seed"],
                max_points=feature_config["max_points"],
            )
        except Exception as error:
            errors[sequence] = f"{type(error).__name__}: {error}"
    return metrics, errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter unified-format sequences with a user-trained model.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--metrics", help="Existing JSONL metrics cache.")
    source.add_argument("--data-root", help="Compute features directly from this unified data root.")
    parser.add_argument("--model", required=True, help="User-trained model bundle. Load only bundles you trust.")
    parser.add_argument("--allowlist", required=True, help="Output text file containing passing sequence names.")
    parser.add_argument("--report", required=True, help="Output JSONL report with pass, reject, and error records.")
    parser.add_argument("--sequence-list")
    args = parser.parse_args()
    if args.metrics and args.sequence_list:
        parser.error("--sequence-list applies only with --data-root")
    return args


def main() -> None:
    args = _parse_args()
    if Path(args.allowlist).resolve() == Path(args.report).resolve():
        raise ValueError("--allowlist and --report must be different files")
    bundle = load_bundle(args.model)
    if args.metrics:
        metrics, errors, feature_config = read_metrics_jsonl(args.metrics)
        if feature_config != bundle["feature_config"]:
            raise ValueError("Metrics cache feature_config does not match the model bundle")
    else:
        metrics, errors = _compute_from_data(args, bundle["feature_config"])
    records = curate_records(bundle, metrics, errors)
    write_allowlist(records, args.allowlist)
    write_jsonl(records, args.report)
    passed = sum(record["status"] == "pass" for record in records)
    rejected = sum(record["status"] == "reject" for record in records)
    print(f"PASS={passed} REJECT={rejected} ERROR={len(errors)}")
    if errors:
        raise SystemExit("Some sequences could not be evaluated; they were excluded from the allowlist")


if __name__ == "__main__":
    main()
