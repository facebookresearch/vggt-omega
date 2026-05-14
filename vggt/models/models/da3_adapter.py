# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from depth_anything_3.cfg import create_object, load_config
from depth_anything_3.registry import MODEL_REGISTRY
from depth_anything_3.utils.geometry import affine_inverse
from vggt.models.utils.pose_enc import extri_intri_to_pose_encoding


class DA3Adapter(nn.Module):
    """
    Adapter that makes Depth Anything 3 compatible with the VGGT trainer/loss pipeline.

    Input:
        batch dict from vggt dataloader (expects key "images")

    Output:
        prediction dict compatible with vggt.losses.loss.MultitaskLoss.
    """

    def __init__(
        self,
        da3_config: Optional[str] = None,
        da3_model_name: str = "da3-large",
        export_feat_layers: Optional[List[int]] = None,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        return_pose_encoding: bool = True,
        pose_encoding_type: str = "absT_quaR_FoV",
        use_input_camera_token: bool = False,
        freeze_unused_modules: bool = True,
        normalize_images_for_da3: bool = True,
        da3_output_extrinsics_convention: str = "w2c",
    ) -> None:
        super().__init__()
        if da3_config is None:
            da3_config = MODEL_REGISTRY[da3_model_name]
        cfg = load_config(da3_config)
        self.model = create_object(cfg)

        self.export_feat_layers = export_feat_layers or []
        self.infer_gs = infer_gs
        self.use_ray_pose = use_ray_pose
        self.ref_view_strategy = ref_view_strategy
        self.return_pose_encoding = return_pose_encoding
        self.pose_encoding_type = pose_encoding_type
        self.use_input_camera_token = use_input_camera_token
        self.freeze_unused_modules = freeze_unused_modules
        self.normalize_images_for_da3 = normalize_images_for_da3
        self.da3_output_extrinsics_convention = da3_output_extrinsics_convention

        self.register_buffer(
            "_imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        if self.freeze_unused_modules:
            self._freeze_unused_parameters()

    def _freeze_unused_parameters(self) -> None:
        # DA3 DualDPT includes auxiliary "*_aux" modules (ray branch). If ray is not used
        # by the loss, these params will be unused under DDP and can cause reduction errors.
        frozen_aux = 0
        for name, param in self.model.named_parameters():
            if "_aux" in name:
                param.requires_grad_(False)
                frozen_aux += 1

        # If camera token injection is disabled, cam_enc is never used in forward.
        frozen_cam_enc = 0
        if not self.use_input_camera_token and hasattr(self.model, "cam_enc") and self.model.cam_enc is not None:
            for param in self.model.cam_enc.parameters():
                if param.requires_grad:
                    param.requires_grad_(False)
                    frozen_cam_enc += 1

        if frozen_aux > 0 or frozen_cam_enc > 0:
            print(
                f"DA3Adapter: froze {frozen_aux} aux params and {frozen_cam_enc} cam_enc params "
                f"(use_input_camera_token={self.use_input_camera_token})"
            )

    @staticmethod
    def _to_3x4(extrinsics: torch.Tensor) -> torch.Tensor:
        if extrinsics.shape[-2:] == (3, 4):
            return extrinsics
        if extrinsics.shape[-2:] == (4, 4):
            return extrinsics[..., :3, :]
        raise ValueError(f"Unsupported extrinsics shape: {tuple(extrinsics.shape)}")

    @staticmethod
    def _to_3x3(intrinsics: torch.Tensor) -> torch.Tensor:
        if intrinsics.shape[-2:] == (3, 3):
            return intrinsics
        if intrinsics.shape[-2:] == (4, 4):
            return intrinsics[..., :3, :3]
        raise ValueError(f"Unsupported intrinsics shape: {tuple(intrinsics.shape)}")

    def _ensure_w2c_extrinsics(self, extrinsics: torch.Tensor) -> torch.Tensor:
        convention = self.da3_output_extrinsics_convention
        if convention == "auto":
            convention = "c2w" if self.use_ray_pose else "w2c"

        if convention == "w2c":
            return extrinsics
        if convention == "c2w":
            return affine_inverse(extrinsics)
        raise ValueError(
            "da3_output_extrinsics_convention must be one of "
            f"['w2c', 'c2w', 'auto'], got: {self.da3_output_extrinsics_convention}"
        )

    def forward(self, batch):
        images = batch["images"]
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if self.normalize_images_for_da3:
            images = (images - self._imagenet_mean.to(images.dtype)) / self._imagenet_std.to(images.dtype)

        # Optional: allow DA3 camera encoder to consume input camera tokens.
        # For pure feed-forward camera regression, keep this disabled.
        if self.use_input_camera_token:
            extrinsics = batch.get("extrinsics", None)
            intrinsics = batch.get("intrinsics", None)
        else:
            extrinsics = None
            intrinsics = None

        output = self.model(
            images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            export_feat_layers=self.export_feat_layers,
            infer_gs=self.infer_gs,
            use_ray_pose=self.use_ray_pose,
            ref_view_strategy=self.ref_view_strategy,
        )

        predictions = {}

        if "depth" in output:
            depth = output["depth"]
            if depth.ndim == 4:
                depth = depth.unsqueeze(-1)
            predictions["depth"] = depth

        if "depth_conf" in output:
            predictions["depth_conf"] = output["depth_conf"]

        if "ray" in output:
            predictions["ray"] = output["ray"]
        if "ray_conf" in output:
            predictions["ray_conf"] = output["ray_conf"]

        if (
            self.return_pose_encoding
            and "extrinsics" in output
            and "intrinsics" in output
        ):
            pred_extrinsics = self._to_3x4(output["extrinsics"])
            pred_extrinsics = self._ensure_w2c_extrinsics(pred_extrinsics)
            pred_intrinsics = self._to_3x3(output["intrinsics"])
            image_hw = images.shape[-2:]
            pose_enc = extri_intri_to_pose_encoding(
                pred_extrinsics,
                pred_intrinsics,
                image_size_hw=image_hw,
                pose_encoding_type=self.pose_encoding_type,
            )
            predictions["pose_enc"] = pose_enc
            predictions["pose_enc_list"] = [pose_enc]

        if not self.training:
            predictions["images"] = images

        return predictions
