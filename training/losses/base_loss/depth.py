# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from losses.base_loss.gradient import gradient_loss_config_wrapper
from losses.unprojection import unproject_depth_to_cam_coords


def _clamp_regression_loss_with_positive_tail_gradient(
    raw_loss: torch.Tensor,
    signed_error: torch.Tensor,
    max_pixel_loss: float,
) -> torch.Tensor:
    hard_clamped_loss = raw_loss.clamp(max=max_pixel_loss)
    positive_tail = (
        (raw_loss.detach() > max_pixel_loss) & (signed_error.detach() > 0)
    ).to(raw_loss.dtype)
    finite_raw_loss = torch.where(
        torch.isfinite(raw_loss),
        raw_loss,
        torch.zeros_like(raw_loss),
    )
    return hard_clamped_loss + positive_tail * (
        finite_raw_loss - finite_raw_loss.detach()
    )


def weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    dim=None,
    keepdim: bool = False,
) -> torch.Tensor:
    weights = weights.to(values.dtype)
    return (values * weights).mean(dim=dim, keepdim=keepdim) / weights.mean(
        dim=dim,
        keepdim=keepdim,
    ).add(1e-7)


def eval_depth(prediction, target):
    """Compute threshold, relative-error, and RMSE depth metrics."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"Depth metric shapes must match, got {prediction.shape} and {target.shape}"
        )
    prediction = prediction.clamp(min=1e-8)
    target = target.clamp(min=1e-8)
    threshold = torch.maximum(target / prediction, prediction / target)
    difference = prediction - target
    log_difference = torch.log(prediction) - torch.log(target)
    return {
        "d1": (threshold < 1.25).float().mean().item(),
        "d2": (threshold < 1.25**2).float().mean().item(),
        "d3": (threshold < 1.25**3).float().mean().item(),
        "abs_rel": (difference.abs() / target).mean().item(),
        "sq_rel": (difference.square() / target).mean().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "rmse_log": log_difference.square().mean().sqrt().item(),
    }


def compute_depth_loss(
    predictions,
    batch,
    weight_normal=1.0,
    min_valid_pts=1000,
    abs_l1_loss_ratio=0.0,
    rel_l1_loss_ratio=1.0,
    max_pixel_loss=100.0,
    use_conf_loss=False,
    conf_gamma=1.0,
    conf_alpha=0.02,
    add_separate_reg_loss=False,
    separate_reg_loss_weight=1.0,
    gradient_loss_config=None,
    apply_aux_loss_to_synthetic_only=False,
):
    """Compute depth regression, confidence, and configured geometry losses."""
    if max_pixel_loss is None or max_pixel_loss <= 0:
        raise ValueError("max_pixel_loss must be positive")
    if use_conf_loss and (
        not add_separate_reg_loss
        or not math.isfinite(separate_reg_loss_weight)
        or separate_reg_loss_weight <= 0
    ):
        raise ValueError(
            "use_conf_loss requires add_separate_reg_loss=True and a positive, finite "
            "separate_reg_loss_weight"
        )
    if use_conf_loss and "depth_conf" not in predictions:
        raise ValueError("use_conf_loss requires predictions['depth_conf']")

    pred_depth = predictions["depth"]
    if pred_depth.ndim == 4:
        pred_depth = pred_depth.unsqueeze(-1)
    gt_depth = batch["depths"]
    if gt_depth.ndim == 4:
        gt_depth = gt_depth.unsqueeze(-1)
    valid_masks = batch["point_masks"]
    intrinsics = batch["intrinsics"]

    valid_seq_mask = batch["valid_seq_mask"] & batch["depth_train_mask"]
    if not valid_seq_mask.any():
        zero = pred_depth.sum() * 0.0
        if use_conf_loss:
            zero = zero + predictions["depth_conf"].sum() * 0.0
        return {
            "loss_depth": zero,
            "loss_pts_l1": zero,
            "loss_reg_depth": zero,
            "depth_l1_unweighted": zero,
            "depth_positive_cap_ratio_max": zero.detach(),
            "loss_pts_l1_max": zero.detach(),
            "loss_normal": zero,
            "loss_conf_depth": zero,
            "depth_conf_median": zero.detach(),
            "d1": 0.0,
            "d2": 0.0,
            "d3": 0.0,
            "abs_rel": 0.0,
            "sq_rel": 0.0,
            "rmse": 0.0,
            "rmse_log": 0.0,
        }

    pred_depth = pred_depth[valid_seq_mask]
    gt_depth = gt_depth[valid_seq_mask]
    intrinsics = intrinsics[valid_seq_mask]
    valid_masks = valid_masks[valid_seq_mask]
    pred_depth_values = pred_depth.squeeze(-1)
    gt_depth_values = gt_depth.squeeze(-1)
    valid_pixels = valid_masks.bool()

    with torch.no_grad():
        metric_mask = valid_pixels & (gt_depth_values > 0)
        if metric_mask.any():
            metrics = eval_depth(
                pred_depth_values[metric_mask],
                gt_depth_values[metric_mask],
            )
        else:
            metrics = {
                "d1": 0.0,
                "d2": 0.0,
                "d3": 0.0,
                "abs_rel": 0.0,
                "sq_rel": 0.0,
                "rmse": 0.0,
                "rmse_log": 0.0,
            }

    pred_local_points = None
    gt_local_points = None
    if gradient_loss_config is not None:
        pred_local_points = unproject_depth_to_cam_coords(pred_depth, intrinsics)
        gt_local_points = unproject_depth_to_cam_coords(gt_depth, intrinsics)

    if valid_pixels.any():
        depth_l1_unweighted = (
            (pred_depth_values - gt_depth_values).abs()[valid_pixels].mean()
        )
    else:
        depth_l1_unweighted = pred_depth.sum() * 0.0

    regression_difference = pred_depth_values - gt_depth_values
    absolute_difference = regression_difference.abs()
    if rel_l1_loss_ratio == 0:
        weighted_difference = absolute_difference * abs_l1_loss_ratio
    else:
        depth_mean = weighted_mean(
            gt_depth_values,
            valid_masks,
            dim=(-2, -1),
            keepdim=True,
        )
        relative_weights = gt_depth_values.clamp_min(0.1 * depth_mean)
        relative_weights = (1.0 / (relative_weights + 1e-6)).clamp(
            min=0.1,
            max=10.0,
        )
        weighted_difference = (
            absolute_difference * relative_weights * rel_l1_loss_ratio
            + absolute_difference * abs_l1_loss_ratio
        )

    if gradient_loss_config is None:
        auxiliary_loss = pred_depth.sum() * 0.0
    elif not apply_aux_loss_to_synthetic_only:
        auxiliary_loss = gradient_loss_config_wrapper(
            pred_local_points.reshape(-1, *pred_local_points.shape[-3:]),
            gt_local_points.reshape(-1, *gt_local_points.shape[-3:]),
            valid_masks.reshape(-1, *valid_masks.shape[-2:]),
            gt_depth=gt_depth,
            config=gradient_loss_config,
        )
    else:
        if "is_synthetic" not in batch:
            raise ValueError(
                "apply_aux_loss_to_synthetic_only=True requires batch['is_synthetic']"
            )
        is_synthetic = torch.as_tensor(
            batch["is_synthetic"],
            device=pred_depth.device,
            dtype=torch.bool,
        )
        if is_synthetic.ndim == 0:
            is_synthetic = is_synthetic.unsqueeze(0)
        is_synthetic = is_synthetic[valid_seq_mask]
        if is_synthetic.any():
            auxiliary_loss = gradient_loss_config_wrapper(
                pred_local_points[is_synthetic].reshape(
                    -1,
                    *pred_local_points.shape[-3:],
                ),
                gt_local_points[is_synthetic].reshape(
                    -1,
                    *gt_local_points.shape[-3:],
                ),
                valid_masks[is_synthetic].reshape(-1, *valid_masks.shape[-2:]),
                gt_depth=gt_depth[is_synthetic],
                config=gradient_loss_config,
            )
        else:
            auxiliary_loss = pred_depth.sum() * 0.0

    confidence_median = pred_depth.new_zeros(())
    if "depth_conf" in predictions:
        confidence_median = predictions["depth_conf"].median()

    positive_cap_ratio_max = pred_depth.new_zeros(())
    if valid_masks.sum() > min_valid_pts:
        regression_per_value = weighted_difference[valid_pixels]
        signed_error = regression_difference[valid_pixels]
        capped_positive_values = regression_per_value.detach().masked_select(
            (regression_per_value.detach() > max_pixel_loss)
            & (signed_error.detach() > 0)
        )
        positive_cap_ratio_max = (
            torch.cat((positive_cap_ratio_max.reshape(1), capped_positive_values)).max()
            / max_pixel_loss
        )

        regression_loss = _clamp_regression_loss_with_positive_tail_gradient(
            regression_per_value,
            signed_error,
            max_pixel_loss,
        ).mean()
        if use_conf_loss:
            depth_confidence = predictions["depth_conf"]
            if depth_confidence.ndim == 5:
                depth_confidence = depth_confidence.squeeze(-1)
            confidence = depth_confidence[valid_seq_mask][valid_pixels]
            confidence_per_pixel = weighted_difference[valid_pixels]
            confidence_loss = (
                (
                    conf_gamma * confidence_per_pixel * confidence
                    - conf_alpha * torch.log(confidence)
                )
                .clamp(max=max_pixel_loss)
                .mean()
            )
            depth_l1_values = weighted_difference[valid_pixels]
            depth_l1_max = depth_l1_values.max()
            depth_l1 = depth_l1_values.mean()
            total_loss = (
                confidence_loss
                + auxiliary_loss * weight_normal
                + regression_loss * separate_reg_loss_weight
            )
        else:
            confidence_loss = 0.0
            depth_l1_max = regression_per_value.clamp(max=max_pixel_loss).max()
            depth_l1 = regression_loss
            total_loss = depth_l1 + auxiliary_loss * weight_normal
    else:
        zero_source = pred_local_points if pred_local_points is not None else pred_depth
        depth_l1 = zero_source.sum() * 0.0
        if use_conf_loss:
            depth_l1 = depth_l1 + predictions["depth_conf"].sum() * 0.0
        depth_l1_max = pred_depth.new_zeros(())
        confidence_loss = 0.0
        regression_loss = depth_l1
        total_loss = depth_l1 + auxiliary_loss * weight_normal

    loss_dict = {
        "loss_depth": total_loss,
        "loss_pts_l1": depth_l1,
        "loss_reg_depth": regression_loss,
        "depth_l1_unweighted": depth_l1_unweighted,
        "depth_positive_cap_ratio_max": positive_cap_ratio_max,
        "loss_pts_l1_max": depth_l1_max,
        "loss_normal": auxiliary_loss,
        "loss_conf_depth": confidence_loss,
        "depth_conf_median": confidence_median,
    }
    loss_dict.update(metrics)
    return loss_dict
