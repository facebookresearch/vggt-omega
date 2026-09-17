from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def pad_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    values = np.asarray(extrinsics, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 4 or values.shape[-2] not in (3, 4):
        raise ValueError(f"expected (N,3,4) or (N,4,4), got {values.shape}")
    if values.shape[-2] == 4:
        return values
    output = np.zeros((values.shape[0], 4, 4), dtype=np.float64)
    output[:, :3] = values
    output[:, 3, 3] = 1.0
    return output


def _sqrt_positive(values: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(values)
    mask = values > 0
    if torch.is_grad_enabled():
        zeros[mask] = torch.sqrt(values[mask])
        return zeros
    return torch.where(mask, torch.sqrt(values), zeros)


def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(matrix.shape)
    shape = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(shape + (9,)), dim=-1
    )
    magnitude = _sqrt_positive(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    candidates = torch.stack(
        [
            torch.stack(
                [magnitude[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1
            ),
            torch.stack(
                [m21 - m12, magnitude[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1
            ),
            torch.stack(
                [m02 - m20, m10 + m01, magnitude[..., 2] ** 2, m12 + m21], dim=-1
            ),
            torch.stack(
                [m10 - m01, m20 + m02, m21 + m12, magnitude[..., 3] ** 2], dim=-1
            ),
        ],
        dim=-2,
    )
    floor = torch.tensor(0.1, dtype=magnitude.dtype, device=magnitude.device)
    candidates = candidates / (2.0 * magnitude[..., None].max(floor))
    quaternion = candidates[
        F.one_hot(magnitude.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(shape + (4,))
    quaternion = quaternion[..., [1, 2, 3, 0]]
    return torch.where(quaternion[..., 3:4] < 0, -quaternion, quaternion)


def _inverse_se3(matrix: torch.Tensor) -> torch.Tensor:
    rotation = matrix[:, :3, :3]
    translation = matrix[:, :3, 3:]
    rotation_t = rotation.transpose(1, 2)
    output = torch.eye(4, dtype=matrix.dtype, device=matrix.device)[None].repeat(
        len(matrix), 1, 1
    )
    output[:, :3, :3] = rotation_t
    output[:, :3, 3:] = -torch.bmm(rotation_t, translation)
    return output


def pose_errors(
    predicted: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    predicted_tensor = torch.from_numpy(pad_extrinsics(predicted)).double()
    target_tensor = torch.from_numpy(pad_extrinsics(target)).double()
    if predicted_tensor.shape != target_tensor.shape:
        raise ValueError(
            f"pose shape mismatch: {predicted_tensor.shape} vs {target_tensor.shape}"
        )
    first, second = torch.combinations(
        torch.arange(len(target_tensor)), 2, with_replacement=False
    ).unbind(-1)
    relative_target = target_tensor[second].bmm(_inverse_se3(target_tensor[first]))
    relative_predicted = predicted_tensor[second].bmm(
        _inverse_se3(predicted_tensor[first])
    )
    target_quaternion = _matrix_to_quaternion(relative_target[:, :3, :3])
    predicted_quaternion = _matrix_to_quaternion(relative_predicted[:, :3, :3])
    rotation_loss = (
        1.0 - (predicted_quaternion * target_quaternion).sum(dim=1) ** 2
    ).clamp(min=1e-15)
    rotation = torch.arccos(1.0 - 2.0 * rotation_loss) * 180.0 / np.pi
    target_translation = relative_target[:, :3, 3]
    predicted_translation = relative_predicted[:, :3, 3]
    target_translation = target_translation / (
        target_translation.norm(dim=1, keepdim=True) + 1e-15
    )
    predicted_translation = predicted_translation / (
        predicted_translation.norm(dim=1, keepdim=True) + 1e-15
    )
    translation_loss = torch.clamp_min(
        1.0 - torch.sum(predicted_translation * target_translation, dim=1) ** 2,
        1e-15,
    )
    translation = torch.acos(torch.sqrt(1.0 - translation_loss)) * 180.0 / np.pi
    translation = torch.minimum(translation, (180.0 - translation).abs())
    return rotation.numpy(), translation.numpy()


def _auc(rotation: np.ndarray, translation: np.ndarray, threshold: int) -> float:
    errors = np.maximum(rotation, translation)
    histogram, _ = np.histogram(errors, bins=np.arange(threshold + 1))
    return float(np.mean(np.cumsum(histogram.astype(float) / len(errors))) * 100.0)


def pose_metrics(rotation: np.ndarray, translation: np.ndarray) -> dict[str, float]:
    rotation = np.asarray(rotation)
    translation = np.asarray(translation)
    if rotation.shape != translation.shape or not rotation.size:
        raise ValueError("pose arrays must be non-empty and have matching shapes")
    return {
        "AUC@3": _auc(rotation, translation, 3),
        "AUC@30": _auc(rotation, translation, 30),
    }


def depth_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if predicted.shape != target.shape:
        raise ValueError(f"depth shape mismatch: {predicted.shape} vs {target.shape}")
    delta = []
    abs_rel = []
    for predicted_frame, target_frame in zip(predicted, target):
        target_frame = target_frame.astype(np.float64, copy=True)
        predicted_frame = predicted_frame.astype(np.float64, copy=True)
        mask = np.isfinite(target_frame) & (target_frame > 0.1) & (target_frame < 100.0)
        if mask.sum() < 10:
            continue
        target_frame = np.clip(target_frame, 0.1, 100.0)
        predicted_frame = np.clip(predicted_frame, 0.1, 100.0)
        target_centered = target_frame[mask] - np.median(target_frame[mask]) + 1e-8
        predicted_centered = (
            predicted_frame[mask] - np.median(predicted_frame[mask]) + 1e-8
        )
        scale = np.median(target_centered / predicted_centered)
        shift = np.median(target_frame[mask] - scale * predicted_frame[mask])
        aligned = np.clip((predicted_frame * scale + shift)[mask], 1e-6, None)
        target_values = target_frame[mask]
        abs_rel.append(float(np.mean(np.abs(aligned - target_values) / target_values)))
        ratio = np.maximum(aligned / target_values, target_values / aligned)
        delta.append(float(np.mean(ratio < 1.25)))
    if not delta:
        raise ValueError("no valid depth frames")
    return {"delta125": float(np.mean(delta)), "AbsRel": float(np.mean(abs_rel))}


def crop_depth(
    depth: np.ndarray, crop: tuple[float, float, float, float]
) -> np.ndarray:
    left, top, right, bottom = crop
    height, width = depth.shape[-2:]
    x0, x1 = round(left * width), round(right * width)
    y0, y1 = round(top * height), round(bottom * height)
    return np.ascontiguousarray(depth[:, y0:y1, x0:x1])


def resize_depth(predicted: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if predicted.shape[1:3] == shape:
        return predicted
    tensor = torch.from_numpy(np.ascontiguousarray(predicted)).float().unsqueeze(1)
    resized = F.interpolate(
        tensor, shape, mode="bilinear", align_corners=False, antialias=True
    )
    return resized.squeeze(1).numpy()
