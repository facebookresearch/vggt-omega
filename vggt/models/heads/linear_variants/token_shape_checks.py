from typing import List, Optional, Tuple

import torch


def get_image_patch_grid_or_raise(
    images: torch.Tensor,
    patch_size: int,
) -> Tuple[int, int, int, int, int]:
    if images.dim() != 5:
        raise ValueError(
            f"images must be [B, S, 3, H, W], got shape {tuple(images.shape)}"
        )

    bsz, seq_len, _, height, width = images.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Image size ({height}, {width}) must be divisible by patch_size ({patch_size})"
        )

    patch_h = height // patch_size
    patch_w = width // patch_size
    num_patches = patch_h * patch_w
    return bsz, seq_len, patch_h, patch_w, num_patches


def get_aggregated_layer_or_raise(
    aggregated_tokens_list: List[torch.Tensor],
    layer_idx: int,
) -> torch.Tensor:
    if layer_idx < -len(aggregated_tokens_list) or layer_idx >= len(aggregated_tokens_list):
        raise ValueError(
            f"intermediate_layer_idx contains out-of-range index {layer_idx} "
            f"for aggregated_tokens_list length {len(aggregated_tokens_list)}"
        )
    return aggregated_tokens_list[layer_idx]


def get_aggregated_patch_tokens_or_raise(
    layer_tokens: torch.Tensor,
    patch_start_idx: int,
    num_patches: int,
    expected_bs_flat: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    if layer_tokens.dim() != 4:
        raise ValueError(
            "aggregated_tokens_list entries must be [B, S, T, C]. "
            f"Got shape {tuple(layer_tokens.shape)}"
        )

    bsz, seq_len, token_len, channels = layer_tokens.shape
    bs_flat = bsz * seq_len
    if expected_bs_flat is not None and bs_flat != expected_bs_flat:
        raise ValueError(
            "aggregated token branch batch mismatch: "
            f"expected {expected_bs_flat}, got {bs_flat}"
        )

    flat_tokens = layer_tokens.reshape(bs_flat, token_len, channels)
    if patch_start_idx < 0:
        raise ValueError(f"patch_start_idx must be >= 0, got {patch_start_idx}")
    if patch_start_idx >= token_len:
        raise ValueError(
            f"patch_start_idx={patch_start_idx} out of range for token_len={token_len}"
        )

    patch_tokens = flat_tokens[:, patch_start_idx:]
    if patch_tokens.shape[1] < num_patches:
        raise ValueError(
            "aggregated token branch has too few patch tokens: "
            f"{patch_tokens.shape[1]} < num_patches({num_patches})"
        )

    return patch_tokens[:, :num_patches, :], bs_flat


def get_patch_embed_tokens_or_raise(
    patch_embed_intermediate: Optional[List[torch.Tensor]],
    expected_bs_flat: int,
    expected_num_patches: int,
    expected_channels: int,
) -> torch.Tensor:
    if patch_embed_intermediate is None:
        raise ValueError(
            "patch_embed_intermediate is required when use_patch_embed_intermediate=True"
        )
    if (
        not isinstance(patch_embed_intermediate, (list, tuple))
        or len(patch_embed_intermediate) != 1
    ):
        raise ValueError(
            "patch_embed_intermediate must be a list/tuple with exactly one tensor."
        )

    layer_tokens = patch_embed_intermediate[0]
    if layer_tokens.dim() != 3:
        raise ValueError(
            "patch_embed_intermediate[0] must be [B*S, T, C]. "
            f"Got shape {tuple(layer_tokens.shape)}"
        )
    if layer_tokens.shape[0] != expected_bs_flat:
        raise ValueError(
            "patch_embed_intermediate[0] batch mismatch: expected "
            f"{expected_bs_flat}, got {layer_tokens.shape[0]}"
        )
    if layer_tokens.shape[1] < expected_num_patches:
        raise ValueError(
            "patch_embed_intermediate[0] has too few tokens: "
            f"{layer_tokens.shape[1]} < num_patches({expected_num_patches})"
        )
    if layer_tokens.shape[-1] != expected_channels:
        raise ValueError(
            "patch_embed_intermediate[0] channel mismatch: "
            f"expected {expected_channels}, got {layer_tokens.shape[-1]}"
        )
    return layer_tokens

