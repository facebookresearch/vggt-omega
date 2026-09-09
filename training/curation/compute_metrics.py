# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN

from . import CURATION_SCHEMA_VERSION, FEATURE_NAMES
from .unified_io import list_sequence_names, load_sequence, validate_relative_name

DEFAULT_SEED = 42
DEFAULT_MAX_POINTS = 100_000
FEATURE_CONFIG_KEYS = {
    "seed",
    "max_points",
    "mask_path",
}


def compute_features(
    scene: dict,
    seed: int = DEFAULT_SEED,
    max_points: int = DEFAULT_MAX_POINTS,
) -> dict[str, float]:
    if len(scene["image_names"]) < 4:
        raise ValueError("At least four frames are required to compute the 38-feature schema")
    if max_points < 100:
        raise ValueError("max_points must be at least 100")

    rng = np.random.default_rng(seed)
    raw_points, colors = _build_point_cloud(scene, max_points, rng)
    if len(raw_points) < 50:
        raise ValueError(f"At least 50 valid depth samples are required, found {len(raw_points)}")

    initial_centroid = np.mean(raw_points, axis=0)
    radii = np.linalg.norm(raw_points - initial_centroid, axis=1)
    keep = radii <= np.percentile(radii, 95)
    raw_points = raw_points[keep]
    colors = colors[keep]
    centroid = np.mean(raw_points, axis=0)
    mean_radius = float(np.mean(np.linalg.norm(raw_points - centroid, axis=1)))
    if not np.isfinite(mean_radius) or mean_radius <= 1e-12:
        raise ValueError("Point cloud has zero spatial extent")
    scale = 1.0 / mean_radius
    points = (raw_points - centroid) * scale

    camera_centers, camera_rotations = _camera_poses(scene["cam_from_worlds"])
    normalized_centers = (camera_centers - centroid) * scale
    depth_cv, completeness, completeness_min = _depth_statistics(scene["depths"])
    trajectory = _trajectory_features(normalized_centers, scene["cam_from_worlds"][:, :, :3])
    local_eigenvalues, local_normals, local_points = _local_geometry(points, rng)
    largest_plane, top_planes = _dominant_plane_coverage(points, rng)
    roughness, roughness_consistency = _multiscale_roughness(points, rng)

    mean_camera = np.mean(normalized_centers, axis=0)
    point_distances = np.linalg.norm(points - mean_camera, axis=1)
    point_depth_cv = _coefficient_of_variation(point_distances)
    global_eigenvalues = np.sort(np.linalg.eigvalsh(np.cov(points, rowvar=False)))[::-1]
    global_linearity = _safe_ratio(global_eigenvalues[0] - global_eigenvalues[1], global_eigenvalues[0])
    local_descending = local_eigenvalues[:, ::-1]
    local_planarity = _safe_column_ratio(
        local_descending[:, 1] - local_descending[:, 2],
        local_descending[:, 0],
    )
    local_linearity = _safe_column_ratio(
        local_descending[:, 0] - local_descending[:, 1],
        local_descending[:, 0],
    )
    local_scattering = _safe_column_ratio(local_descending[:, 2], local_descending[:, 0])
    structured = (
        ((local_planarity > 0.7) & (local_linearity < 0.3))
        | ((local_linearity > 0.7) & (local_planarity < 0.3))
    ) & (local_scattering <= 0.25)

    up_vectors = np.einsum("nij,j->ni", camera_rotations, np.array([0.0, -1.0, 0.0]))
    mean_up = np.mean(up_vectors, axis=0)
    if np.linalg.norm(mean_up) <= 1e-12:
        up_consistency = 180.0
    else:
        mean_up /= np.linalg.norm(mean_up)
        up_angles = np.degrees(np.arccos(np.clip(up_vectors @ mean_up, -1.0, 1.0)))
        up_consistency = float(np.std(up_angles))
    up_velocity = _rotation_angles_between_vectors(up_vectors[:-1], up_vectors[1:])

    focal_ratios = []
    widths = []
    for image, intrinsic in zip(scene["images"], scene["intrinsics"]):
        height, width = image.shape[:2]
        widths.append(width)
        focal_ratios.append(float((intrinsic[0, 0] + intrinsic[1, 1]) / (2 * max(width, height))))

    values = {
        "Camera_UpVector_Consistency_value": up_consistency,
        "Convex_Hull_Volume_Ratio_value": _convex_hull_ratio(points),
        "Depth_Map_Coeff_of_Variation_value": depth_cv,
        "Depth_Map_Completeness_Minimum_value": completeness_min,
        "Depth_Map_Completeness_value": completeness,
        "Dominant_Plane_Coverage_value": largest_plane,
        "Dominant_Plane_TopK_Coverage_value": top_planes,
        "Focal_Length_Statistics_value": float(np.mean(focal_ratios)),
        "Frame_Dimension_Statistics_value": float(np.median(widths)),
        "Free_Space_Violation_Score_value": _free_space_score(points, scene, centroid, scale, rng),
        "Local_Curvature_Distribution_value": _curvature_entropy(local_eigenvalues),
        "Mean_Point_Distance_Before_Normalization_value": mean_radius,
        "Multiscale_Roughness_Scale_Consistency_value": roughness_consistency,
        "Multiscale_Surface_Roughness_value": roughness,
        "Normalization_Scale_Factor_value": scale,
        "Parallax_Angle_Analysis_value": _parallax_angle(raw_points, scene, camera_centers, rng),
        "Point_Cloud_Color_Diversity_value": _color_diversity(colors, rng),
        "Point_Cloud_Connectivity_value": _connectivity(points),
        "Point_Cloud_Density_Uniformity_value": _density_uniformity(points, rng),
        "Point_Cloud_Depth_Standard_Deviation_value": point_depth_cv,
        "Point_Cloud_Noise_Level_value": _noise_ratio(points, rng),
        "Point_Cloud_PCA_Shape_Analysis_value": global_linearity,
        "Point_Cloud_Planarity_value": float(np.mean(local_planarity)),
        "Point_Cloud_Spectral_Analysis_value": _spectral_ratio(points, rng),
        "Surface_Local_Structure_Score_value": float(np.mean(structured)),
        "Surface_Normal_Consistency_value": _normal_consistency(local_points, local_normals),
        "Total_Rotational_Motion_value": trajectory["rotation_mean"],
        "Trajectory_Aspect_Ratio_value": _trajectory_aspect_ratio(normalized_centers),
        "Trajectory_Baseline_value": float(np.median(np.linalg.norm(np.diff(normalized_centers, axis=0), axis=1))),
        "Trajectory_Consistency_value": trajectory["acceleration"],
        "Trajectory_Jerk_value": trajectory["jerk"],
        "Trajectory_Rotational_Acceleration_Max_value": trajectory["rotation_acceleration_max"],
        "Trajectory_Rotational_Jerk_Max_value": trajectory["rotation_jerk_max"],
        "Trajectory_Translational_Acceleration_Max_value": trajectory["translation_acceleration_max"],
        "Trajectory_Translational_Jerk_Max_value": trajectory["translation_jerk_max"],
        "UpVector_Angular_Velocity_Max_value": float(np.max(up_velocity)),
        "UpVector_Angular_Velocity_Mean_value": float(np.mean(up_velocity)),
        "View_Overlap__Covisibility_Score_value": _pose_overlap(normalized_centers, camera_rotations),
    }
    return validate_features(values)


def validate_features(features: object) -> dict[str, float]:
    if not isinstance(features, dict):
        raise ValueError("features must be a JSON object")
    missing = sorted(set(FEATURE_NAMES) - set(features))
    extra = sorted(set(features) - set(FEATURE_NAMES))
    if missing or extra:
        raise ValueError(f"Feature schema mismatch; missing={missing}, extra={extra}")
    ordered = {}
    for name in FEATURE_NAMES:
        value = features[name]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"Feature {name} must be numeric, got {type(value).__name__}")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"Feature {name} is not finite: {value}")
        ordered[name] = value
    return ordered


def make_feature_config(
    seed: int,
    max_points: int,
    mask_path: str | None,
) -> dict:
    return validate_feature_config(
        {
            "seed": seed,
            "max_points": max_points,
            "mask_path": mask_path,
        }
    )


def validate_feature_config(config: object) -> dict:
    if not isinstance(config, dict) or set(config) != FEATURE_CONFIG_KEYS:
        raise ValueError(f"feature_config keys must be {sorted(FEATURE_CONFIG_KEYS)}")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise ValueError("feature_config.seed must be an integer")
    if (
        not isinstance(config["max_points"], int)
        or isinstance(config["max_points"], bool)
        or config["max_points"] < 100
    ):
        raise ValueError("feature_config.max_points must be an integer of at least 100")
    mask_path = config["mask_path"]
    if mask_path is not None and (not isinstance(mask_path, str) or not mask_path):
        raise ValueError("feature_config.mask_path must be a non-empty string or null")
    if mask_path is not None:
        validate_relative_name(mask_path, "feature_config.mask_path")
    return {key: config[key] for key in sorted(FEATURE_CONFIG_KEYS)}


def read_metrics_jsonl(path: str | Path) -> tuple[dict[str, dict[str, float]], dict[str, str], dict]:
    metrics = {}
    errors = {}
    feature_config = None
    for line_number, raw_line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if record.get("schema_version") != CURATION_SCHEMA_VERSION:
            raise ValueError(f"{path}:{line_number}: unsupported schema_version")
        record_feature_config = validate_feature_config(record.get("feature_config"))
        if feature_config is None:
            feature_config = record_feature_config
        elif record_feature_config != feature_config:
            raise ValueError(f"{path}:{line_number}: feature_config differs from earlier records")
        sequence = record.get("sequence")
        if not isinstance(sequence, str) or not sequence:
            raise ValueError(f"{path}:{line_number}: sequence must be a non-empty string")
        validate_relative_name(sequence, f"{path}:{line_number}: sequence")
        if sequence in metrics or sequence in errors:
            raise ValueError(f"{path}:{line_number}: duplicate sequence {sequence!r}")
        status = record.get("status")
        if status == "ok":
            metrics[sequence] = validate_features(record.get("features"))
        elif status == "error" and isinstance(record.get("error"), str):
            errors[sequence] = record["error"]
        else:
            raise ValueError(f"{path}:{line_number}: status must be 'ok' or a valid 'error' record")
    if not metrics and not errors:
        raise ValueError(f"Metrics file is empty: {path}")
    return metrics, errors, feature_config


def write_jsonl(records: list[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def _build_point_cloud(scene: dict, max_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    valid_counts = [
        int(np.count_nonzero((depth > 0) & (depth < 1e6) & np.isfinite(depth)))
        for depth in scene["depths"]
    ]
    total_valid = sum(valid_counts)
    all_points = []
    all_colors = []
    for image, depth, extrinsic, intrinsic, valid_count in zip(
        scene["images"],
        scene["depths"],
        scene["cam_from_worlds"],
        scene["intrinsics"],
        valid_counts,
    ):
        valid_indices = np.flatnonzero((depth > 0) & (depth < 1e6) & np.isfinite(depth))
        frame_limit = valid_count
        if total_valid > max_points:
            frame_limit = max(1, int(np.ceil(max_points * valid_count / total_valid)))
        if len(valid_indices) > frame_limit:
            valid_indices = np.sort(rng.choice(valid_indices, frame_limit, replace=False))
        if not len(valid_indices):
            continue
        height, width = depth.shape
        rows, columns = np.unravel_index(valid_indices, (height, width))
        z = depth[rows, columns].astype(np.float64)
        x = (columns.astype(np.float64) + 0.5 - intrinsic[0, 2]) * z / intrinsic[0, 0]
        y = (rows.astype(np.float64) + 0.5 - intrinsic[1, 2]) * z / intrinsic[1, 1]
        camera_points = np.column_stack((x, y, z))
        rotation = extrinsic[:, :3]
        translation = extrinsic[:, 3]
        world_points = (camera_points - translation) @ rotation
        all_points.append(world_points)
        all_colors.append(image[rows, columns])
    if not all_points:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)
    points = np.concatenate(all_points)
    colors = np.concatenate(all_colors)
    if len(points) > max_points:
        indices = np.sort(rng.choice(len(points), max_points, replace=False))
        points = points[indices]
        colors = colors[indices]
    return points, colors


def _camera_poses(extrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotations = extrinsics[:, :, :3]
    translations = extrinsics[:, :, 3]
    camera_rotations = np.transpose(rotations, (0, 2, 1))
    centers = -np.einsum("nij,nj->ni", camera_rotations, translations)
    return centers, camera_rotations


def _depth_statistics(depths: list[np.ndarray]) -> tuple[float, float, float]:
    coefficients = []
    completeness = []
    for depth in depths:
        valid = depth[(depth > 0) & (depth < 1e6) & np.isfinite(depth)]
        completeness.append(float(len(valid) / depth.size))
        if len(valid) >= 10:
            low, high = np.percentile(valid, [1, 99])
            clipped = valid[(valid >= low) & (valid <= high)]
            coefficients.append(_coefficient_of_variation(clipped))
    if not coefficients:
        raise ValueError("Sampled depth maps do not contain enough valid values")
    return float(np.mean(coefficients)), float(np.mean(completeness)), float(np.min(completeness))


def _trajectory_features(centers: np.ndarray, rotations: np.ndarray) -> dict[str, float]:
    translation_acceleration = np.diff(centers, n=2, axis=0)
    translation_jerk = np.diff(centers, n=3, axis=0)
    relative = Rotation.from_matrix(rotations[1:]) * Rotation.from_matrix(rotations[:-1]).inv()
    angular_velocity = relative.as_rotvec()
    rotation_acceleration = np.diff(angular_velocity, axis=0)
    rotation_jerk = np.diff(angular_velocity, n=2, axis=0)
    translation_acceleration_scores = np.sum(translation_acceleration**2, axis=1)
    translation_jerk_scores = np.sum(translation_jerk**2, axis=1)
    rotation_acceleration_scores = np.sum(rotation_acceleration**2, axis=1)
    rotation_jerk_scores = np.sum(rotation_jerk**2, axis=1)
    rotation_angles = np.linalg.norm(angular_velocity, axis=1)
    return {
        "acceleration": float(np.mean(translation_acceleration_scores) + np.mean(rotation_acceleration_scores)),
        "jerk": float(np.mean(translation_jerk_scores) + np.mean(rotation_jerk_scores)),
        "translation_acceleration_max": float(np.max(translation_acceleration_scores)),
        "translation_jerk_max": float(np.max(translation_jerk_scores)),
        "rotation_acceleration_max": float(np.max(rotation_acceleration_scores)),
        "rotation_jerk_max": float(np.max(rotation_jerk_scores)),
        "rotation_mean": float(np.degrees(np.mean(rotation_angles))),
    }


def _local_geometry(
    points: np.ndarray,
    rng: np.random.Generator,
    sample_limit: int = 2_000,
    neighbors: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_points = _sample_rows(points, sample_limit, rng)
    neighbor_count = min(neighbors, len(sample_points))
    if neighbor_count < 3:
        raise ValueError("Point cloud is too small for local geometry")
    tree = cKDTree(sample_points)
    _, indices = tree.query(sample_points, k=neighbor_count)
    eigenvalues = []
    normals = []
    for neighborhood_indices in indices:
        neighborhood = sample_points[neighborhood_indices]
        centered = neighborhood - np.mean(neighborhood, axis=0)
        values, vectors = np.linalg.eigh(centered.T @ centered / max(len(centered) - 1, 1))
        eigenvalues.append(np.maximum(values, 0.0))
        normals.append(vectors[:, 0])
    return np.asarray(eigenvalues), np.asarray(normals), sample_points


def _dominant_plane_coverage(points: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    remaining = _sample_rows(points, 5_000, rng)
    total = len(remaining)
    coverages = []
    for _ in range(3):
        if len(remaining) < 3:
            break
        best = np.zeros(len(remaining), dtype=bool)
        for _ in range(1_000):
            sample = remaining[rng.choice(len(remaining), 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = np.linalg.norm(normal)
            if norm <= 1e-12:
                continue
            normal /= norm
            inliers = np.abs((remaining - sample[0]) @ normal) <= 0.02
            if np.count_nonzero(inliers) > np.count_nonzero(best):
                best = inliers
        count = int(np.count_nonzero(best))
        if not count:
            break
        coverages.append(count / total)
        remaining = remaining[~best]
    return (max(coverages, default=0.0), float(np.sum(coverages)))


def _multiscale_roughness(points: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = []
    for voxel_size in (0.01, 0.05, 0.1, 0.2):
        scaled = _voxel_downsample(points, voxel_size)
        scaled = _sample_rows(scaled, 1_250, rng)
        if len(scaled) < 3:
            values.append(0.0)
            continue
        _, indices = cKDTree(scaled).query(scaled, k=min(30, len(scaled)))
        roughness = []
        for neighborhood_indices in indices:
            neighborhood = scaled[neighborhood_indices]
            centered = neighborhood - np.mean(neighborhood, axis=0)
            eigenvalues = np.linalg.eigvalsh(centered.T @ centered / max(len(centered) - 1, 1))
            roughness.append(np.sqrt(max(float(eigenvalues[0]), 0.0)))
        values.append(float(np.mean(roughness)))
    ratio = _safe_ratio(values[0], values[-1], default=1.0)
    mean_roughness = float(np.mean(values))
    consistency = max(0.0, 1.0 - _coefficient_of_variation(np.asarray(values))) if mean_roughness > 1e-9 else 0.0
    return ratio, consistency


def _convex_hull_ratio(points: np.ndarray) -> float:
    try:
        hull_volume = ConvexHull(points).volume
    except QhullError:
        return 0.0
    box_volume = float(np.prod(np.ptp(points, axis=0)))
    return _safe_ratio(hull_volume, box_volume)


def _curvature_entropy(eigenvalues: np.ndarray) -> float:
    variation = _safe_column_ratio(eigenvalues[:, 0], np.sum(eigenvalues, axis=1))
    counts, _ = np.histogram(variation, bins=50)
    probabilities = counts[counts > 0] / np.sum(counts)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _normal_consistency(points: np.ndarray, normals: np.ndarray) -> float:
    _, indices = cKDTree(points).query(points, k=min(20, len(points)))
    scores = []
    for index, neighborhood_indices in enumerate(indices):
        alignments = np.abs(normals[neighborhood_indices] @ normals[index])
        scores.append(float(np.mean(alignments)))
    return float(np.median(scores))


def _connectivity(points: np.ndarray) -> float:
    downsampled = _voxel_downsample(points, 0.05)
    if len(downsampled) > 10_000:
        downsampled = downsampled[:10_000]
    labels = DBSCAN(eps=0.1, min_samples=10).fit_predict(downsampled)
    _, counts = np.unique(labels[labels >= 0], return_counts=True)
    return float(np.max(counts) / np.sum(counts)) if len(counts) else 0.0


def _density_uniformity(points: np.ndarray, rng: np.random.Generator) -> float:
    samples = _sample_rows(points, 5_000, rng)
    distances, _ = cKDTree(points).query(samples, k=min(11, len(points)))
    return _coefficient_of_variation(distances[:, -1])


def _noise_ratio(points: np.ndarray, rng: np.random.Generator) -> float:
    samples = _sample_rows(points, 10_000, rng)
    distances, _ = cKDTree(samples).query(samples, k=min(21, len(samples)))
    means = np.mean(distances[:, 1:], axis=1)
    threshold = np.mean(means) + 2.0 * np.std(means)
    return float(np.mean(means > threshold))


def _color_diversity(colors: np.ndarray, rng: np.random.Generator) -> float:
    sample = _sample_rows(colors, 5_000, rng).astype(np.float32) / 255.0
    lab = cv2.cvtColor(sample.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return float(np.mean(np.std(lab, axis=0)))


def _pose_overlap(centers: np.ndarray, rotations: np.ndarray) -> float:
    distance_limit = 0.25 * np.linalg.norm(np.ptp(centers, axis=0))
    if distance_limit <= 1e-12:
        distance_limit = 1.0
    forward = rotations[:, :, 2]
    counts = []
    for index in range(len(centers)):
        distances = np.linalg.norm(centers - centers[index], axis=1)
        directions = forward @ forward[index]
        matches = (distances <= distance_limit) & (directions >= np.cos(np.radians(45.0)))
        counts.append((np.count_nonzero(matches) - 1) / (len(centers) - 1))
    return float(np.mean(counts))


def _parallax_angle(
    points: np.ndarray,
    scene: dict,
    camera_centers: np.ndarray,
    rng: np.random.Generator,
) -> float:
    points = _sample_rows(points, 500, rng)
    angles = []
    for point in points:
        visible = []
        for index, (image, extrinsic, intrinsic) in enumerate(
            zip(scene["images"], scene["cam_from_worlds"], scene["intrinsics"])
        ):
            camera_point = extrinsic[:, :3] @ point + extrinsic[:, 3]
            if camera_point[2] <= 0:
                continue
            projected = intrinsic @ camera_point
            column, row = projected[:2] / projected[2]
            height, width = image.shape[:2]
            if 0 <= column < width and 0 <= row < height:
                visible.append(index)
        if len(visible) < 2:
            continue
        visible_centers = camera_centers[visible]
        pairwise = np.linalg.norm(visible_centers[:, None] - visible_centers[None, :], axis=2)
        first, second = np.unravel_index(np.argmax(pairwise), pairwise.shape)
        ray_a = point - visible_centers[first]
        ray_b = point - visible_centers[second]
        cosine = _safe_ratio(float(ray_a @ ray_b), float(np.linalg.norm(ray_a) * np.linalg.norm(ray_b)))
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    return float(np.median(angles)) if angles else 0.0


def _free_space_score(
    points: np.ndarray,
    scene: dict,
    centroid: np.ndarray,
    scale: float,
    rng: np.random.Generator,
) -> float:
    tree = cKDTree(points)
    violations = []
    frames = zip(scene["depths"][:10], scene["cam_from_worlds"][:10], scene["intrinsics"][:10])
    for depth, extrinsic, intrinsic in frames:
        valid = np.flatnonzero((depth > 0) & (depth < 1e6) & np.isfinite(depth))
        if not len(valid):
            continue
        selected = rng.choice(valid, min(100, len(valid)), replace=False)
        rows, columns = np.unravel_index(selected, depth.shape)
        camera_center = -extrinsic[:, :3].T @ extrinsic[:, 3]
        normalized_center = (camera_center - centroid) * scale
        for row, column in zip(rows, columns):
            z = float(depth[row, column])
            camera_point = np.array(
                [
                    (column + 0.5 - intrinsic[0, 2]) * z / intrinsic[0, 0],
                    (row + 0.5 - intrinsic[1, 2]) * z / intrinsic[1, 1],
                    z,
                ]
            )
            target = (extrinsic[:, :3].T @ (camera_point - extrinsic[:, 3]) - centroid) * scale
            segment = target - normalized_center
            length = np.linalg.norm(segment)
            if length <= 1e-12:
                continue
            direction = segment / length
            query = normalized_center + np.linspace(0.05, 0.9, 18)[:, None] * segment
            candidate_lists = tree.query_ball_point(query, r=0.01)
            candidate_indices = {item for group in candidate_lists for item in group}
            found = False
            for index in candidate_indices:
                along = float((points[index] - normalized_center) @ direction)
                if 0.1 < along < 0.9 * length:
                    found = True
                    break
            violations.append(found)
    return float(np.mean(violations)) if violations else 0.0


def _spectral_ratio(points: np.ndarray, rng: np.random.Generator) -> float:
    points = _sample_rows(points, 1_000, rng)
    neighbor_count = min(16, len(points))
    distances, indices = cKDTree(points).query(points, k=neighbor_count)
    rows = np.repeat(np.arange(len(points)), neighbor_count - 1)
    columns = indices[:, 1:].reshape(-1)
    local_scale = np.median(distances[:, 1:], axis=1)
    denominator = np.repeat(np.maximum(2.0 * local_scale**2, 1e-12), neighbor_count - 1)
    weights = np.exp(-(distances[:, 1:].reshape(-1) ** 2) / denominator)
    adjacency = csr_matrix((weights, (rows, columns)), shape=(len(points), len(points)))
    adjacency = (adjacency + adjacency.T) * 0.5
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    normalized = adjacency.multiply(inverse_sqrt[:, None]).multiply(inverse_sqrt[None, :])
    laplacian = eye(len(points), format="csr") - normalized
    count = min(25, len(points) - 1)
    if count < 10:
        return 100.0
    eigenvalues = np.sort(
        eigsh(laplacian, k=count, which="SM", v0=np.ones(len(points)), return_eigenvectors=False)
    )
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) < 10:
        return 100.0
    midpoint = len(eigenvalues) // 2
    return min(100.0, _safe_ratio(np.sum(eigenvalues[midpoint:]), np.sum(eigenvalues[:midpoint]), 100.0))


def _trajectory_aspect_ratio(centers: np.ndarray) -> float:
    eigenvalues = np.sort(np.linalg.eigvalsh(np.cov(centers, rowvar=False)))[::-1]
    return min(1e9, _safe_ratio(eigenvalues[0], eigenvalues[1], 1e9))


def _rotation_angles_between_vectors(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    products = np.sum(first * second, axis=1)
    norms = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return np.degrees(np.arccos(np.clip(products / np.maximum(norms, 1e-12), -1.0, 1.0)))


def _voxel_downsample(points: np.ndarray, size: float) -> np.ndarray:
    cells = np.floor(points / size).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(indices)]


def _sample_rows(array: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if len(array) <= limit:
        return array
    return array[np.sort(rng.choice(len(array), limit, replace=False))]


def _coefficient_of_variation(values: np.ndarray) -> float:
    return _safe_ratio(float(np.std(values)), float(np.mean(values)))


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if abs(denominator) > 1e-12 else default


def _safe_column_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=output, where=np.abs(denominator) > 1e-12)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute curation features from unified-format sequences.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True, help="Output JSONL metrics cache.")
    parser.add_argument("--sequence-list")
    parser.add_argument("--mask-path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = []
    error_count = 0
    names = list_sequence_names(args.data_root, args.sequence_list)
    feature_config = make_feature_config(
        args.seed,
        args.max_points,
        args.mask_path,
    )
    for sequence_name in names:
        try:
            scene = load_sequence(
                args.data_root,
                sequence_name,
                mask_path=args.mask_path,
            )
            features = compute_features(scene, seed=args.seed, max_points=args.max_points)
            record = {
                "schema_version": CURATION_SCHEMA_VERSION,
                "sequence": sequence_name,
                "status": "ok",
                "feature_config": feature_config,
                "features": features,
            }
        except Exception as error:
            error_count += 1
            record = {
                "schema_version": CURATION_SCHEMA_VERSION,
                "sequence": sequence_name,
                "status": "error",
                "feature_config": feature_config,
                "error": f"{type(error).__name__}: {error}",
            }
        records.append(record)
        print(f"{sequence_name}: {record['status'].upper()}")
    write_jsonl(records, args.output)
    if error_count:
        raise SystemExit(f"Feature extraction failed for {error_count} of {len(names)} sequences")


if __name__ == "__main__":
    main()
