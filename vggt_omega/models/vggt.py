import contextlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vggt_omega.models.aggregator import Aggregator
from vggt_omega.models.heads import CameraHeadLinear, DPTLinearHead
from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer


def _make_dinov3_vitl16() -> DinoVisionTransformer:
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="max",
        pos_embed_rope_dtype="fp32",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-5,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    )
    model.init_weights()
    return model


def _checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


class VGGTOmega(nn.Module):
    """Minimal VGGT-Omega inference model for camera and depth prediction."""

    def __init__(self) -> None:
        super().__init__()

        self.patch_size = 16
        self.pose_encoding_type = "absT_quaR_FoV"

        self.patch_embed = _make_dinov3_vitl16()
        self.aggregator = Aggregator(
            num_register_tokens=16,
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            use_qk_norm=True,
            global_attn_mode="specials",
            global_attn_indices=[2, 6, 9, 14, 20],
            aa_order=["frame", "global"],
            use_dino_clsreg=False,
            rope_dtype="fp32",
            rope_normalize_coords="max",
            disable_rope_global=True,
            use_checkpoint=False,
        )
        self.camera_head = CameraHeadLinear(
            dim_in=2048,
            pose_encoding_type=self.pose_encoding_type,
            mlp_ratio=[0.5],
            patch_size=16,
            extra_attention_depth=4,
            extra_attention_dim=-1,
            extra_attention_pre_norm=True,
            extra_attention_post_norm=True,
            extra_attention_use_qk_norm=False,
            use_checkpoint=False,
            disable_last_layer_amp=True,
        )
        self.depth_head = DPTLinearHead(
            dim_in=2048,
            output_dim=2,
            activation="exp",
            conf_activation="expp1",
            patch_size=16,
            features=256,
            out_channels=[256, 512, 1024, 1024],
            intermediate_layer_idx=[4, 11, 17, 23],
            predict_mask=False,
            mask_activation="none",
            pos_embed=True,
            disable_last_layer_amp=True,
            proj_type="linear",
            mlp_ratio=0.5,
            fusion_block_relu_inplace=False,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
        **kwargs: Any,
    ) -> "VGGTOmega":
        model = cls(**kwargs)
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        state_dict = _checkpoint_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=strict)
        return model

    @staticmethod
    def _amp_context(images: torch.Tensor):
        if not images.is_cuda:
            return contextlib.nullcontext()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def forward(self, images: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if isinstance(images, dict):
            images = images["images"]

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if len(images.shape) != 5:
            raise ValueError(f"Expected images with shape [S,3,H,W] or [B,S,3,H,W], got {tuple(images.shape)}")
        if images.shape[2] != 3:
            raise ValueError(f"Expected RGB images with 3 channels, got {images.shape[2]}")

        with self._amp_context(images):
            aggregated_tokens_list, patch_start_idx, patch_embed_intermediate = self.aggregator(
                images, self.patch_embed
            )

        if patch_embed_intermediate is not None:
            raise RuntimeError("VGGT-Omega inference does not use patch_embed_intermediate")

        predictions = {}
        head_context = torch.autocast(device_type="cuda", enabled=False) if images.is_cuda else contextlib.nullcontext()
        with head_context:
            pose_enc_list = self.camera_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
            )
            predictions["pose_enc"] = pose_enc_list[-1]
            predictions["pose_enc_list"] = pose_enc_list

            depth, depth_conf, depth_mask = self.depth_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                patch_embed_intermediate=patch_embed_intermediate,
            )
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf
            if depth_mask is not None:
                predictions["depth_mask"] = depth_mask

        if not self.training:
            predictions["images"] = images
        return predictions
