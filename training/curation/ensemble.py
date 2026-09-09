# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib.metadata
import json
import os
from pathlib import Path

import joblib
import numpy as np

from . import CURATION_SCHEMA_VERSION, FEATURE_NAMES
from .compute_metrics import validate_feature_config, validate_features

MODEL_NAMES = (
    "catboost",
    "extra_trees",
    "xgboost",
    "hist_gradient_boosting",
    "lightgbm",
    "random_forest",
)

MODEL_PARAMETER_NAMES = {
    "catboost": {"auto_class_weights", "depth", "iterations", "learning_rate"},
    "extra_trees": {"class_weight", "max_depth", "max_features", "n_estimators"},
    "hist_gradient_boosting": {"class_weight", "learning_rate", "max_iter", "max_leaf_nodes"},
    "lightgbm": {"class_weight", "learning_rate", "n_estimators", "num_leaves"},
    "random_forest": {"class_weight", "max_depth", "max_features", "n_estimators"},
    "xgboost": {"learning_rate", "max_depth", "n_estimators", "scale_pos_weight"},
}


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path is not None else Path(__file__).with_name("ensemble_config.json")
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {config_path}: {error}") from error
    _validate_config(config)
    return config


def build_estimators(config: dict, labels: np.ndarray) -> list[tuple[str, object]]:
    _validate_config(config)
    labels = np.asarray(labels, dtype=bool)
    positive_count = int(np.count_nonzero(labels))
    negative_count = int(len(labels) - positive_count)
    if not positive_count or not negative_count:
        raise ValueError("Training labels must contain both True and False")

    try:
        from catboost import CatBoostClassifier
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
        from xgboost import XGBClassifier
    except ImportError as error:
        raise ImportError('Install the curation dependencies with pip install -e ".[curation]"') from error

    seed = config["random_state"]
    parameters = config["models"]
    catboost = parameters["catboost"]
    extra_trees = parameters["extra_trees"]
    xgboost = parameters["xgboost"]
    histogram = parameters["hist_gradient_boosting"]
    lightgbm = parameters["lightgbm"]
    random_forest = parameters["random_forest"]
    if xgboost["scale_pos_weight"] != "auto":
        raise ValueError("xgboost.scale_pos_weight must be 'auto'")

    return [
        (
            "catboost",
            CatBoostClassifier(
                auto_class_weights=catboost["auto_class_weights"],
                depth=catboost["depth"],
                iterations=catboost["iterations"],
                learning_rate=catboost["learning_rate"],
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
            ),
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                class_weight=extra_trees["class_weight"],
                max_depth=extra_trees["max_depth"],
                max_features=extra_trees["max_features"],
                n_estimators=extra_trees["n_estimators"],
                random_state=seed,
            ),
        ),
        (
            "xgboost",
            XGBClassifier(
                learning_rate=xgboost["learning_rate"],
                max_depth=xgboost["max_depth"],
                n_estimators=xgboost["n_estimators"],
                scale_pos_weight=negative_count / positive_count,
                random_state=seed,
                eval_metric="logloss",
            ),
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                class_weight=histogram["class_weight"],
                learning_rate=histogram["learning_rate"],
                max_iter=histogram["max_iter"],
                max_leaf_nodes=histogram["max_leaf_nodes"],
                random_state=seed,
            ),
        ),
        (
            "lightgbm",
            LGBMClassifier(
                class_weight=lightgbm["class_weight"],
                learning_rate=lightgbm["learning_rate"],
                n_estimators=lightgbm["n_estimators"],
                num_leaves=lightgbm["num_leaves"],
                random_state=seed,
                verbosity=-1,
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                class_weight=random_forest["class_weight"],
                max_depth=random_forest["max_depth"],
                max_features=random_forest["max_features"],
                n_estimators=random_forest["n_estimators"],
                random_state=seed,
            ),
        ),
    ]


def fit_estimators(
    estimators: list[tuple[str, object]],
    features: np.ndarray,
    labels: np.ndarray,
) -> list[tuple[str, object]]:
    _validate_estimator_names(estimators)
    return [(name, estimator.fit(features, labels)) for name, estimator in estimators]


def predict_two_false(models: list[tuple[str, object]], features: np.ndarray) -> dict[str, np.ndarray]:
    _validate_estimator_names(models)
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected an N x {len(FEATURE_NAMES)} feature matrix, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Feature matrix contains non-finite values")

    predictions_by_model = []
    scores_by_model = []
    for name, model in models:
        prediction = np.asarray(model.predict(matrix), dtype=bool)
        score = _positive_scores(model, matrix)
        if prediction.shape != (len(matrix),):
            raise ValueError(f"Model {name} returned prediction shape {prediction.shape}")
        if score.shape != (len(matrix),) or not np.all(np.isfinite(score)):
            raise ValueError(f"Model {name} returned invalid positive scores")
        predictions_by_model.append(prediction)
        scores_by_model.append(score)
    predictions = np.stack(predictions_by_model)
    false_votes = np.sum(~predictions, axis=0)
    scores = np.mean(np.stack(scores_by_model), axis=0)
    return {
        "accepted": false_votes < 2,
        "false_votes": false_votes,
        "positive_score": scores,
        "base_predictions": predictions.T,
    }


def feature_matrix(records: dict[str, dict[str, float]], names: list[str]) -> np.ndarray:
    rows = []
    for name in names:
        if name not in records:
            raise ValueError(f"No metrics found for labeled sequence {name!r}")
        values = validate_features(records[name])
        rows.append([values[feature_name] for feature_name in FEATURE_NAMES])
    return np.asarray(rows, dtype=np.float64)


def make_bundle(
    models: list[tuple[str, object]],
    config: dict,
    feature_config: dict,
    training_summary: dict,
) -> dict:
    _validate_estimator_names(models)
    _validate_config(config)
    _validate_training_summary(training_summary)
    return {
        "schema_version": CURATION_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "model_names": list(MODEL_NAMES),
        "models": models,
        "config": config,
        "feature_config": validate_feature_config(feature_config),
        "training_summary": training_summary,
        "dependencies": _dependency_versions(),
    }


def save_bundle(bundle: dict, path: str | Path) -> None:
    validate_bundle(bundle)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    joblib.dump(bundle, temporary, compress=3)
    with temporary.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def load_bundle(path: str | Path) -> dict:
    bundle = joblib.load(path)
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: object) -> None:
    if not isinstance(bundle, dict):
        raise ValueError("Model bundle must be a dictionary")
    expected = {
        "schema_version",
        "feature_names",
        "model_names",
        "models",
        "config",
        "feature_config",
        "training_summary",
        "dependencies",
    }
    if set(bundle) != expected:
        raise ValueError(f"Model bundle keys must be {sorted(expected)}, got {sorted(bundle)}")
    if bundle["schema_version"] != CURATION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported model schema version: {bundle['schema_version']}")
    if tuple(bundle["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Model bundle uses a different feature schema")
    if tuple(bundle["model_names"]) != MODEL_NAMES:
        raise ValueError("Model bundle uses a different model set")
    _validate_estimator_names(bundle["models"])
    _validate_config(bundle["config"])
    validate_feature_config(bundle["feature_config"])
    _validate_training_summary(bundle["training_summary"])
    if not isinstance(bundle["dependencies"], dict):
        raise ValueError("Model dependency metadata must be a dictionary")


def _validate_config(config: object) -> None:
    if not isinstance(config, dict):
        raise ValueError("Ensemble config must be a JSON object")
    expected = {"schema_version", "random_state", "validation_split", "models"}
    if set(config) != expected:
        raise ValueError(f"Ensemble config keys must be {sorted(expected)}, got {sorted(config)}")
    if config["schema_version"] != CURATION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported config schema version: {config['schema_version']}")
    if not isinstance(config["random_state"], int) or isinstance(config["random_state"], bool):
        raise ValueError("random_state must be an integer")
    if not isinstance(config["validation_split"], (int, float)) or not 0 < config["validation_split"] < 1:
        raise ValueError("validation_split must be between 0 and 1")
    models = config["models"]
    if not isinstance(models, dict) or set(models) != set(MODEL_NAMES):
        raise ValueError(f"models must contain exactly {list(MODEL_NAMES)}")
    for name, expected_parameters in MODEL_PARAMETER_NAMES.items():
        if not isinstance(models[name], dict) or set(models[name]) != expected_parameters:
            raise ValueError(f"{name} parameters must be {sorted(expected_parameters)}")
    if models["xgboost"]["scale_pos_weight"] != "auto":
        raise ValueError("xgboost.scale_pos_weight must be 'auto'")


def _validate_estimator_names(estimators: object) -> None:
    if not isinstance(estimators, list) or len(estimators) != len(MODEL_NAMES):
        raise ValueError(f"Expected {len(MODEL_NAMES)} fitted models")
    names = []
    for item in estimators:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("Each model must be stored as a (name, estimator) tuple")
        names.append(item[0])
    if tuple(names) != MODEL_NAMES:
        raise ValueError(f"Model order must be {list(MODEL_NAMES)}, got {names}")


def _validate_training_summary(summary: object) -> None:
    expected = {"sample_count", "positive_count", "negative_count", "xgboost_scale_pos_weight", "validation"}
    if not isinstance(summary, dict) or set(summary) != expected:
        raise ValueError(f"training_summary keys must be {sorted(expected)}")
    counts = (summary["sample_count"], summary["positive_count"], summary["negative_count"])
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in counts):
        raise ValueError("Training sample counts must be positive integers")
    if summary["positive_count"] + summary["negative_count"] != summary["sample_count"]:
        raise ValueError("Training positive and negative counts do not sum to sample_count")
    ratio = summary["xgboost_scale_pos_weight"]
    if not isinstance(ratio, (int, float)) or not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("xgboost_scale_pos_weight must be finite and positive")
    if not isinstance(summary["validation"], dict):
        raise ValueError("training_summary.validation must be a dictionary")


def _positive_scores(model: object, features: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        return np.asarray(model.predict(features), dtype=np.float64)
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1:
        raise ValueError(f"Model classes do not contain a unique positive class: {classes.tolist()}")
    return probabilities[:, positive[0]]


def _dependency_versions() -> dict[str, str]:
    versions = {}
    for distribution in (
        "catboost",
        "joblib",
        "lightgbm",
        "numpy",
        "opencv-python",
        "scikit-learn",
        "scipy",
        "xgboost",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    return versions
