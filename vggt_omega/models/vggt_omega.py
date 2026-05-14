import contextlib
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vggt_omega.models.aggregator import Aggregator
from vggt_omega.models.heads import CameraHead, DenseHead, TextAlignmentHead
from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer


checkpoint_key_renames = [
    ("depth_head.", "dense_head."),
    ("camera_head.extra_attention_pre_norm.", "camera_head.token_norm."),
    ("camera_head.extra_attention_blocks.", "camera_head.trunk."),
    ("camera_head.token_norm.", "camera_head.trunk_norm."),
    ("alignment_head.student.pre_norm.", "alignment_head.student.token_norm."),
    ("alignment_head.student.extra_attention_blocks.", "alignment_head.student.trunk."),
    ("alignment_head.student.token_norm.", "alignment_head.student.trunk_norm."),
]


class VGGTOmega(nn.Module):
    """Minimal VGGT-Omega inference model for camera and depth prediction."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_alignment: bool = False,
    ) -> None:
        super().__init__()

        self.patch_embed = _build_patch_embed(patch_size=patch_size, embed_dim=embed_dim)
        self.aggregator = Aggregator(patch_size=patch_size, embed_dim=embed_dim)
        _warn_if_rope_not_max(self.patch_embed, self.aggregator)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.dense_head = DenseHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_depth else None
        self.alignment_head = TextAlignmentHead(dim_in=2 * embed_dim) if enable_alignment else None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
        enable_alignment: bool | None = None,
        **kwargs: Any,
    ) -> "VGGTOmega":
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        state_dict = rename_state_dict_keys(_checkpoint_state_dict(checkpoint), checkpoint_key_renames)
        if enable_alignment is None:
            enable_alignment = any(key.startswith("alignment_head.student.") for key in state_dict)
        model = cls(enable_alignment=enable_alignment, **kwargs)
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
            aggregated_tokens_list, patch_start_idx = self.aggregator(images, self.patch_embed)

        predictions = {}
        head_context = torch.autocast(device_type="cuda", enabled=False) if images.is_cuda else contextlib.nullcontext()
        with head_context:
            if self.camera_head is not None:
                predictions["pose_enc"] = self.camera_head(
                    aggregated_tokens_list,
                    patch_start_idx=patch_start_idx,
                )

            if self.dense_head is not None:
                depth, depth_conf = self.dense_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.alignment_head is not None:
                predictions.update(
                    self.alignment_head(
                        aggregated_tokens_list,
                        patch_start_idx=patch_start_idx,
                    )
                )

        if not self.training:
            predictions["images"] = images
        return predictions


def _build_patch_embed(patch_size: int, embed_dim: int) -> DinoVisionTransformer:
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=patch_size,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="max",
        pos_embed_rope_dtype="fp32",
        embed_dim=embed_dim,
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


def rename_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
    rules: list[tuple[str, str]],
) -> dict[str, torch.Tensor]:
    renamed = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in rules:
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix) :]
                break
        if new_key in renamed:
            raise ValueError(f"Checkpoint key rename collision: {key!r} maps to existing key {new_key!r}")
        renamed[new_key] = value
    return renamed


def _warn_if_rope_not_max(patch_embed: nn.Module, aggregator: nn.Module) -> None:
    for name, module in (("patch_embed", patch_embed), ("aggregator", aggregator)):
        rope_embed = getattr(module, "rope_embed", None)
        normalize_coords = getattr(rope_embed, "normalize_coords", None)
        if normalize_coords != "max":
            warnings.warn(
                f"{name} RoPE normalize_coords is {normalize_coords!r}; "
                "the released VGGT-Omega checkpoint was trained with 'max'.",
                stacklevel=2,
            )
