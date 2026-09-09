# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .compute_metrics import read_metrics_jsonl
from .ensemble import (
    build_estimators,
    feature_matrix,
    fit_estimators,
    load_config,
    make_bundle,
    predict_two_false,
    save_bundle,
)
from .unified_io import validate_relative_name


def read_labels(path: str | Path) -> dict[str, bool]:
    labels = {}
    for line_number, raw_line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2 or fields[1].lower() not in {"true", "false"}:
            raise ValueError(f"{path}:{line_number}: expected '<sequence> True|False'")
        sequence, value = fields
        validate_relative_name(sequence, f"{path}:{line_number}: sequence")
        if sequence in labels:
            raise ValueError(f"{path}:{line_number}: duplicate sequence {sequence!r}")
        labels[sequence] = value.lower() == "true"
    if not labels:
        raise ValueError(f"Labels file is empty: {path}")
    return labels


def train_model(
    metrics_path: str | Path,
    labels_path: str | Path,
    config_path: str | Path | None = None,
) -> tuple[dict, dict]:
    metrics, metric_errors, feature_config = read_metrics_jsonl(metrics_path)
    labels_by_name = read_labels(labels_path)
    failed_labels = sorted(set(labels_by_name) & set(metric_errors))
    if failed_labels:
        raise ValueError(f"Feature extraction failed for labeled sequences: {failed_labels}")
    missing = sorted(set(labels_by_name) - set(metrics))
    if missing:
        raise ValueError(f"No metrics found for labeled sequences: {missing}")

    names = list(labels_by_name)
    features = feature_matrix(metrics, names)
    labels = np.asarray([labels_by_name[name] for name in names], dtype=bool)
    if np.unique(labels).size != 2:
        raise ValueError("Labels must contain both True and False")

    config = load_config(config_path)
    try:
        train_features, validation_features, train_labels, validation_labels = train_test_split(
            features,
            labels,
            test_size=config["validation_split"],
            random_state=config["random_state"],
            stratify=labels,
        )
    except ValueError as error:
        raise ValueError(f"Cannot create the configured stratified validation split: {error}") from error

    validation_models = fit_estimators(build_estimators(config, train_labels), train_features, train_labels)
    validation_predictions = predict_two_false(validation_models, validation_features)["accepted"]
    evaluation = {
        "samples": len(labels),
        "training_samples": len(train_labels),
        "validation_samples": len(validation_labels),
        "accuracy": float(accuracy_score(validation_labels, validation_predictions)),
        "precision": float(precision_score(validation_labels, validation_predictions, zero_division=0)),
        "recall": float(recall_score(validation_labels, validation_predictions, zero_division=0)),
        "f1": float(f1_score(validation_labels, validation_predictions, zero_division=0)),
    }

    final_models = fit_estimators(build_estimators(config, labels), features, labels)
    positive_count = int(np.count_nonzero(labels))
    training_summary = {
        "sample_count": len(labels),
        "positive_count": positive_count,
        "negative_count": len(labels) - positive_count,
        "xgboost_scale_pos_weight": (len(labels) - positive_count) / positive_count,
        "validation": evaluation,
    }
    return make_bundle(final_models, config, feature_config, training_summary), evaluation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the six-model curation ensemble from user labels.")
    parser.add_argument("--metrics", required=True, help="JSONL produced by curation.compute_metrics.")
    parser.add_argument("--labels", required=True, help="Text file with '<sequence> True|False' lines.")
    parser.add_argument("--output", required=True, help="Output model bundle. Load only bundles you trust.")
    parser.add_argument("--config")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    bundle, evaluation = train_model(args.metrics, args.labels, args.config)
    save_bundle(bundle, args.output)
    print(json.dumps(evaluation, indent=2, sort_keys=True))
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
