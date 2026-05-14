from typing import List, Optional


def validate_head_core_config_or_raise(
    intermediate_layer_idx: List[int],
    use_patch_embed_intermediate: bool,
    multiscale_norm_type: str,
    fusion_mode: str,
    dpt_fusion_features: int,
    fuser_output_scale: int,
) -> None:
    if not intermediate_layer_idx:
        raise ValueError(
            "intermediate_layer_idx must be provided for the multiscale token path."
        )

    if use_patch_embed_intermediate and len(intermediate_layer_idx) != 3:
        raise ValueError(
            "When use_patch_embed_intermediate=True, intermediate_layer_idx must "
            "contain exactly 3 layers so patch_embed + aggregated branches form 4 "
            f"fusion inputs. Got {len(intermediate_layer_idx)}."
        )

    if multiscale_norm_type not in ("shared", "per_level"):
        raise ValueError(
            f"Unknown multiscale_norm_type: {multiscale_norm_type}. "
            "Supported: ['shared', 'per_level']"
        )

    supported_fusion_modes = ("dpt", "gated_progressive")
    if fusion_mode not in supported_fusion_modes:
        raise ValueError(
            f"Unknown fusion_mode: {fusion_mode}. "
            f"Supported: {list(supported_fusion_modes)}"
        )

    if dpt_fusion_features <= 0:
        raise ValueError(
            f"dpt_fusion_features must be > 0, got {dpt_fusion_features}"
        )
    if fuser_output_scale <= 0:
        raise ValueError(
            f"fuser_output_scale must be > 0, got {fuser_output_scale}"
        )
    if fusion_mode == "dpt" and fuser_output_scale != 4:
        raise ValueError(
            "fusion_mode='dpt' currently outputs 1/4-resolution fused features, "
            f"so fuser_output_scale must be 4 (got {fuser_output_scale})."
        )
    if fusion_mode == "gated_progressive" and fuser_output_scale not in (1, 4):
        raise ValueError(
            "fusion_mode='gated_progressive' currently supports fused output scales "
            f"1 (keep patch-grid resolution) or 4 (quarter resolution), got {fuser_output_scale}."
        )


def resolve_multiscale_out_channels_or_raise(
    multiscale_out_channels: Optional[List[int]],
    dim_in: int,
    num_levels: int,
) -> List[int]:
    if multiscale_out_channels is None:
        default_channels = max(dim_in // num_levels, 1)
        multiscale_out_channels = [default_channels] * num_levels

    if len(multiscale_out_channels) != num_levels:
        raise ValueError(
            "multiscale_out_channels length must match intermediate_layer_idx length. "
            f"Got {len(multiscale_out_channels)} vs {num_levels}"
        )
    return multiscale_out_channels


def resolve_patch_embed_out_channels_or_raise(
    use_patch_embed_intermediate: bool,
    patch_embed_dim: Optional[int],
    patch_embed_out_channels: Optional[int],
    dim_in: int,
    num_aggregated_levels: int,
) -> Optional[int]:
    if not use_patch_embed_intermediate:
        return None

    if patch_embed_dim is None or patch_embed_dim <= 0:
        raise ValueError(
            "patch_embed_dim must be > 0 when use_patch_embed_intermediate=True, "
            f"got {patch_embed_dim}"
        )

    if patch_embed_out_channels is None:
        patch_embed_out_channels = max(dim_in // num_aggregated_levels, 1)

    if patch_embed_out_channels <= 0:
        raise ValueError(
            "patch_embed_out_channels must be > 0 when patch-embed branch is enabled, "
            f"got {patch_embed_out_channels}"
        )
    return patch_embed_out_channels


def build_fusion_input_channels_or_raise(
    multiscale_out_channels: List[int],
    use_patch_embed_intermediate: bool,
    patch_embed_out_channels: Optional[int],
    fusion_mode: str,
) -> List[int]:
    fusion_input_channels = list(multiscale_out_channels)
    if use_patch_embed_intermediate:
        if patch_embed_out_channels is None:
            raise ValueError(
                "patch_embed_out_channels must be set when patch-embed branch is enabled."
            )
        fusion_input_channels = [
            patch_embed_out_channels,
            *fusion_input_channels,
        ]

    if fusion_mode == "dpt":
        if len(fusion_input_channels) != 4:
            raise ValueError(
                "fusion_mode='dpt' requires exactly 4 multiscale branches "
                f"(got {len(fusion_input_channels)})."
            )
    elif fusion_mode == "gated_progressive":
        if len(fusion_input_channels) not in (1, 4):
            raise ValueError(
                "fusion_mode='gated_progressive' supports either 1 branch "
                "(single-layer path) or 4 branches (progressive fusion), "
                f"got {len(fusion_input_channels)}."
            )
    else:
        raise ValueError(
            f"Unknown fusion_mode: {fusion_mode}. Supported: ['dpt', 'gated_progressive']"
        )
    return fusion_input_channels


def get_final_shuffle_factor_or_raise(
    output_dim: int,
    patch_size: int,
    fuser_output_scale: int,
) -> int:
    if output_dim <= 1:
        raise ValueError(f"output_dim must be > 1, got {output_dim}")
    if patch_size % fuser_output_scale != 0:
        raise ValueError(
            f"patch_size ({patch_size}) must be divisible by {fuser_output_scale} "
            "to pixel-shuffle from fused feature scale."
        )
    return patch_size // fuser_output_scale

