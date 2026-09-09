# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from losses.base_loss.camera import PairwisePoseLoss, compute_camera_loss
from losses.base_loss.consistency import compute_geometric_consistency_loss
from losses.base_loss.depth import compute_depth_loss
from losses.base_loss.point import compute_point_loss
from losses.gt_normalization import normalize_gt_batch
from losses.normalization import normalize_predictions_differentiable


def _extract_weight(config):
    if config is None:
        return None, 0.0
    config = dict(config)
    weight = config.pop("weight")
    return config, weight


class MultitaskLoss(torch.nn.Module):
    """Weighted sum of the camera, depth, point and track-consistency losses.

        loss_objective = camera.weight * loss_camera
                 + depth.weight  * loss_depth
                 + point.weight  * loss_point
                 + track.weight  * loss_consistency

    Points are not predicted by a head: they come from unprojecting predicted depth with the
    predicted camera, so loss_point couples the camera and depth predictions.
    """

    def __init__(
        self,
        camera=None,
        depth=None,
        point=None,
        track=None,
        normalize_predictions=False,
        normalize_gt=None,
        rel_to_first_cam=False,
        intrinsics_warmup_ratio=0.5,
    ):
        super().__init__()
        self.camera, self.camera_weight = _extract_weight(camera)
        self.depth, self.depth_weight = _extract_weight(depth)
        self.point, self.point_weight = _extract_weight(point)
        self.track, self.track_weight = _extract_weight(track)
        self.normalize_predictions = normalize_predictions
        self.normalize_gt = normalize_gt
        self.rel_to_first_cam = rel_to_first_cam
        self.intrinsics_warmup_ratio = intrinsics_warmup_ratio

        pairwise_weight_rotation = None
        pairwise_weight_translation = None
        if self.camera is not None:
            pairwise_weight_rotation = self.camera.pop("pairwise_weight_rotation", 1.0)
            pairwise_weight_translation = self.camera.pop(
                "pairwise_weight_translation", 1.0
            )
        if self.camera is not None and self.camera.get("weight_pairwise", 0.0) > 0:
            self.pairwise_loss = PairwisePoseLoss(
                weight_rotation=pairwise_weight_rotation,
                weight_translation=pairwise_weight_translation,
            )
        else:
            self.pairwise_loss = None

    def forward(self, predictions, batch, schedule_progress) -> dict:
        """`schedule_progress` is the fraction of training completed, in [0, 1). It drives the
        intrinsics warmup, so it must be the same value the LR schedulers see."""
        if self.normalize_gt:
            batch = normalize_gt_batch(batch, normalization_type=self.normalize_gt)

        total_loss = 0
        loss_dict = {}

        if "depth" in predictions:
            loss_dict["mean_depth_raw"] = predictions["depth"].mean().item()

        if self.normalize_predictions:
            # Decides whether the model has to learn absolute scale or is free to pick its own.
            with torch.autocast(device_type="cuda", enabled=False):
                predictions, batch = normalize_predictions_differentiable(
                    predictions,
                    batch,
                    schedule_progress,
                    rel_to_first_cam=self.rel_to_first_cam,
                    intrinsics_warmup_ratio=self.intrinsics_warmup_ratio,
                )

        if "pose_enc_list" in predictions:
            camera_loss_dict = compute_camera_loss(
                predictions,
                batch,
                pairwise_loss_fn=self.pairwise_loss,
                **self.camera,
            )
            total_loss = (
                total_loss + camera_loss_dict["loss_camera"] * self.camera_weight
            )
            loss_dict.update(camera_loss_dict)

        if "depth" in predictions:
            depth_loss_dict = compute_depth_loss(predictions, batch, **self.depth)
            total_loss = total_loss + depth_loss_dict["loss_depth"] * self.depth_weight
            loss_dict.update(depth_loss_dict)

        if (
            self.point is not None
            and "depth" in predictions
            and "pose_enc_list" in predictions
        ):
            predictions, batch, point_loss_dict = compute_point_loss(
                predictions,
                batch,
                intrinsics_warmup_ratio=self.intrinsics_warmup_ratio,
                schedule_progress=schedule_progress,
                **self.point,
            )
            total_loss = total_loss + point_loss_dict["loss_point"] * self.point_weight
            loss_dict.update(point_loss_dict)

        if (
            self.track is not None
            and "world_points" in predictions
            and "tracks" in batch
        ):
            consistency_loss_dict = compute_geometric_consistency_loss(
                predictions, batch, **self.track
            )
            total_loss = (
                total_loss
                + consistency_loss_dict["loss_consistency"] * self.track_weight
            )
            loss_dict.update(consistency_loss_dict)

        loss_dict["loss_objective"] = total_loss
        return loss_dict
