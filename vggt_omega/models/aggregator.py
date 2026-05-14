# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from functools import partial
from typing import Optional, Tuple, List, Any
import math
import logging
from vggt_omega.models.layers import SelfAttentionBlock, RopePositionEmbedding, Mlp, LayerScale, PatchEmbed, RMSNorm, SwiGLUFFN
from vggt_omega.models.layers.utils import named_apply

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}


def init_weights_vit(module: nn.Module, name: str = "") -> None:
    """
    Initialize common layers with ViT-style defaults.
    - Linear: truncated normal weights (std=0.02), zero bias
    - LayerNorm/LayerScale/PatchEmbed/RMSNorm: call their reset_parameters
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


class Aggregator(nn.Module):
    """
    Alternating-attention encoder over video frames, as in VGGT.

    Applies attention in two passes per depth step:
    - "frame" pass attends within each frame independently
    - "global" pass mixes information across frames; the participating tokens
      are controlled by `global_attn_mode`
    Special tokens (camera + register) are always kept in front of each frame,
    and RoPE is applied only to patch tokens.

    Args:
        patch_size (int): Patch size used by the patch embedder.
        embed_dim (int): Token embedding dimension.
        depth (int): Number of alternating-attention steps.
        num_heads (int): Attention heads per block.
        mlp_ratio (float): Expansion ratio for the MLP/FFN.
        num_register_tokens (int): Number of per-frame register tokens.
        block_fn (Any): Block constructor called as block_fn(...)->nn.Module.
        qkv_bias (bool): Add bias to QKV projections.
        proj_bias (bool): Add bias to output projection.
        ffn_bias (bool): Add bias to MLP layers.
        aa_order (List[str]): Order of passes for each step, e.g. ["frame","global"].
        aa_block_size (int): Blocks per pass before switching. Must be 1 (current impl).
        use_qk_norm (bool): Whether the attention blocks use QK normalization.
        rope_freq (int | None): Base freq for RoPE. None or <=0 disables RoPE.
        rope_rescale_coords (int): RoPE coordinate rescaling factor.
        rope_normalize_coords (str): RoPE coordinate normalization mode.
        init_values (float): Initial scale for LayerScale.
        global_attn_mode (str): Global attention mode. One of "all", "specials", "registers".
            If not "all", see `global_attn_partial_ratio`.
            The layers are distributed evenly. global_attn_partial_ratio = 0.0 means all layers use "all" mode.
        patch_residual_last_layer (bool): If True, add a residual connection from the original
            patch tokens (from patch_embed_model) to the patch portion of frame_intermediates and
            global_intermediates at the last layer output. Only patch tokens are affected, not
            camera_token or register_token.
        last_n_global_specials (int): If > 0, override the last N global attention layers to use
            "specials" attention mode. Default is -1 (disabled).
        mask_k_bias (bool): Whether QKV projections use DINOv3's masked K bias.
    """

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 16,
        block_fn: Any = SelfAttentionBlock,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        ffn_layer: str = "mlp",
        aa_order: List[str] = ["frame", "global"],
        aa_block_size: int = 1,
        use_qk_norm: bool = True,
        rope_freq: Optional[int] = 100,
        rope_dtype: str = "fp32",
        rope_rescale_coords: float | None = None,
        rope_normalize_coords: str = "max",
        match_patch_embed_rope_behavior: bool = False,
        init_values: float = 1e-5,
        global_attn_mode: str = "specials",
        global_attn_partial_ratio: float = 1.0,
        global_attn_indices: Optional[List[int]] = [2, 6, 9, 14, 20],
        use_dino_clsreg: bool = False,
        disable_rope_global: bool = True,
        force_global_attn_specials: bool = False,
        patch_residual_last_layer: bool = False,
        last_n_global_specials: int = -1,
        mask_k_bias: bool = True,
    ):
        super().__init__()

        self.disable_rope_global = disable_rope_global
        self.match_patch_embed_rope_behavior = match_patch_embed_rope_behavior
        if not disable_rope_global:
            raise RuntimeError("global RoPE is not supported now")
        self.rope_dtype = dtype_dict[rope_dtype]

        logging.info(f"Aggregator: RoPE normalize_coords={rope_normalize_coords}")
        if self.match_patch_embed_rope_behavior:
            logging.info("Aggregator: matching patch_embed RoPE behavior (per-block sampling)")

        if ffn_layer not in ffn_layer_dict:
             raise ValueError(f"Unknown ffn_layer: {ffn_layer}")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]

        # Initialize rotary position embedding if enabled
        self.rope_embed = (
            RopePositionEmbedding(
                embed_dim=embed_dim,                  # default is 1024
                num_heads=num_heads,                # default is 16
                base=rope_freq,                      # default is 100
                rescale_coords=rope_rescale_coords,     # default is None
                normalize_coords=rope_normalize_coords,  # default is 'separate'
                dtype=self.rope_dtype,
            )
            if rope_freq is not None and rope_freq > 0
            else None
        )

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio = mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    ffn_layer=ffn_layer_cls,
                    init_values=init_values,
                    use_qk_norm=use_qk_norm,
                    mask_k_bias=mask_k_bias,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio = mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    ffn_layer=ffn_layer_cls,
                    init_values=init_values,
                    use_qk_norm=use_qk_norm,
                    mask_k_bias=mask_k_bias,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        self.use_qk_norm = use_qk_norm
        self.use_dino_clsreg = use_dino_clsreg
        self.patch_residual_last_layer = patch_residual_last_layer

        if self.aa_block_size != 1:
            raise ValueError("aa_block_size must be 1 for now")

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        if self.use_dino_clsreg:
            self.patch_start_idx += 5 # 1 for dino cls token, 4 for dino register tokens

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        if force_global_attn_specials:
            self.global_attn_modes = ["specials" for _ in range(depth)]
        else:
            if global_attn_mode not in ["all", "specials", "registers"]:
                raise ValueError(f"Unknown global_attn_mode: {global_attn_mode}")

            self.global_attn_modes = []
            if global_attn_mode != "all":
                if global_attn_indices is not None:
                    self.global_attn_modes = ["all"] * depth
                    for idx in global_attn_indices:
                        if 0 <= idx < depth:
                            self.global_attn_modes[idx] = global_attn_mode
                else:
                    if global_attn_partial_ratio < 0.0 or global_attn_partial_ratio > 1.0:
                        raise ValueError(f"global_attn_partial_ratio must be in [0, 1], but got {global_attn_partial_ratio}")

                    for i in range(depth):
                        if math.floor((i+1)*global_attn_partial_ratio) > math.floor(i*global_attn_partial_ratio):
                            self.global_attn_modes.append(global_attn_mode)
                        else:
                            self.global_attn_modes.append("all")
            else:
                self.global_attn_modes = [global_attn_mode for _ in range(depth)]

            # Ensure the first, second, and last global attention blocks are "all" mode
            self.global_attn_modes[0] = "all"
            self.global_attn_modes[1] = "all"
            self.global_attn_modes[-1] = "all"

        # Override the last N global layers to use "specials" attention if specified
        if last_n_global_specials > 0:
            for i in range(max(0, depth - last_n_global_specials), depth):
                self.global_attn_modes[i] = "specials"
            logging.info(f"Aggregator: Set last {last_n_global_specials} global layers to 'specials' attention")

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize learnable parameters and positional encodings."""
        if self.rope_embed is not None:
            self.rope_embed._init_weights()

        nn.init.normal_(self.camera_token, std=1e-3)
        nn.init.normal_(self.register_token, std=1e-3)
        named_apply(init_weights_vit, self)

    def forward(
        self, 
        images: torch.Tensor, 
        patch_embed_model: nn.Module,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Run the alternating-attention encoder.

        Args:
            images: Tensor of shape [B, S, 3, H, W] with values in [0, 1].
            patch_embed_model: Module that maps images to patch tokens.
                It may return either a tensor of shape [B*S, P, C] or a dict
                containing "x_norm_patchtokens".
        Returns:
            - outputs: list of tensors, each of shape [B, S, P, 2 * C], where
              P = 1 + num_register_tokens + HW and the last dimension is the
              concatenation of the per-frame and global representations from
              the same depth step.
            - patch_start_idx: index where patch tokens start (i.e., after specials).
        """
        B, S, C_in, H, W = images.shape

        

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)

        # Expand camera and register tokens to match batch size and sequence length
        # All the first frame in a sequence will have the same camera token
        # All the other frames in a sequence will have the same camera token
        # (1, 2, 1, embed_dim) -> (B, S, 1, embed_dim)
        # (1, 2, num_register_tokens, embed_dim) -> (B, S, num_register_tokens, embed_dim)
        
        
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)


        patch_tokens = patch_embed_model(images)
        
        if isinstance(patch_tokens, dict):
            if self.use_dino_clsreg:
                dino_cls_token = patch_tokens["x_norm_clstoken"]
                dino_reg_token = patch_tokens["x_storage_tokens"]
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        if self.use_dino_clsreg:
            register_token = torch.cat([register_token, dino_cls_token[:, None], dino_reg_token], dim=1)

        _, P, C = patch_tokens.shape

        # Save patch_tokens for potential residual connection at last layer
        patch_tokens_for_residual = patch_tokens.view(B, S, -1, C) if self.patch_residual_last_layer else None

        # TODO: add absolute position embedding

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        # Shape convention (per-frame layout during the frame pass):
        # - Special tokens (camera + registers) come first within each frame.
        # - RoPE should rotate only the last HW patch tokens per frame (specials untouched).
        #   Let ps = patch_start_idx = 1 + num_register_tokens and HW = (H/patch_size)*(W/patch_size).
        #   Then P = ps + HW and the first ps tokens are the prefix excluded from RoPE.

        rope_frame = None
        rope_global = None
        rope_hw = (H // self.patch_size, W // self.patch_size)
        if self.rope_embed is not None and not self.match_patch_embed_rope_behavior:
            # Precompute RoPE for the frame pass and match dtype/device to tokens.
            # Shapes:
            #   rope_frame:  (HW, D) applies per frame to the last HW patch tokens only.
            # SelfAttention.apply_rope computes prefix = N - rope_len and leaves
            # the first 'prefix' tokens (the specials) unrotated.
            rope_dtype = self.rope_dtype
            rope_device = patch_tokens.device

            with torch.no_grad():
                rope_sin, rope_cos = self.rope_embed(H=rope_hw[0], W=rope_hw[1])
                rope_frame = (rope_sin.to(device=rope_device, dtype=rope_dtype), rope_cos.to(device=rope_device, dtype=rope_dtype))

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []
        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, rope_sincos=rope_frame, rope_hw=rope_hw
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx,
                        global_attn_mode=self.global_attn_modes[global_idx],
                        rope_sincos=rope_global
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            assert len(frame_intermediates) == len(global_intermediates)
            is_last_layer = (frame_idx == self.depth)
            for i in range(len(frame_intermediates)):
                frame_inter = frame_intermediates[i]
                global_inter = global_intermediates[i]

                # Add patch residual connection at the last layer (only to patch tokens)
                if self.patch_residual_last_layer and is_last_layer:
                    ps = self.patch_start_idx
                    # frame_inter and global_inter: [B, S, P, C], patch_tokens_for_residual: [B, S, HW, C]
                    # Use torch.cat instead of in-place assignment to preserve autograd graph
                    frame_inter = torch.cat([
                        frame_inter[:, :, :ps, :],
                        frame_inter[:, :, ps:, :] + patch_tokens_for_residual
                    ], dim=2)
                    global_inter = torch.cat([
                        global_inter[:, :, :ps, :],
                        global_inter[:, :, ps:, :] + patch_tokens_for_residual
                    ], dim=2)

                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_inter, global_inter], dim=-1)
                output_list.append(concat_inter)

        return output_list, self.patch_start_idx

    def _process_frame_attention(
        self,
        tokens: torch.Tensor,
        B: int,
        S: int,
        P: int,
        C: int,
        frame_idx: int,
        rope_sincos: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rope_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
        """
        Apply per-frame attention blocks.
        Tokens are in shape (B*S, P, C) where specials precede patches.
        With RoPE length HW, SelfAttention.apply_rope uses prefix = P - HW = patch_start_idx,
        rotating only the last HW patch tokens per frame.
        """
        # Ensure canonical frame shape (views only)
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            current_rope = rope_sincos
            if self.match_patch_embed_rope_behavior and self.rope_embed is not None:
                if rope_hw is None:
                    raise RuntimeError("rope_hw must be provided when match_patch_embed_rope_behavior=True")
                rope_sin, rope_cos = self.rope_embed(H=rope_hw[0], W=rope_hw[1])
                current_rope = (
                    rope_sin.to(device=tokens.device, dtype=self.rope_dtype),
                    rope_cos.to(device=tokens.device, dtype=self.rope_dtype),
                )
            tokens = self.frame_blocks[frame_idx](tokens, current_rope)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(
        self,
        tokens: torch.Tensor,
        B: int,
        S: int,
        P: int,
        C: int,
        global_idx: int,
        global_attn_mode: str,
        rope_sincos: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
        """
        Apply global attention blocks across frames.

        Global attention expects no RoPE. In "all" mode we flatten directly to
        (B, S*P, C). The partial modes repack specials/patches so the attention
        block only sees the requested subset. On return, tokens are restored to
        per-frame layout (B, S, P, C).
        """
        assert rope_sincos is None, "_process_global_attention expects rope_sincos=None"

        ps = self.patch_start_idx
        tokens = tokens.view(B, S, P, C)

        intermediates = []

        if global_attn_mode == "all":
            attn_tokens = tokens.view(B, S * P, C)
        else:
            special = tokens[:, :, :ps, :].reshape(B, S * ps, C)
            patches = tokens[:, :, ps:, :].reshape(B, S * (P - ps), C)
            camera_toks_per_frame = None

        if global_attn_mode == "registers":
            # camera tokens are at index 0 of special tokens
            num_registers = ps - 1
            special_per_frame = special.view(B, S, ps, C)
            camera_toks_per_frame = special_per_frame[:, :, 0:1, :]
            register_toks_per_frame = special_per_frame[:, :, 1:ps, :]
            attn_tokens = register_toks_per_frame.reshape(B, S * num_registers, C)
        elif global_attn_mode == "specials":
            attn_tokens = special
            
        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            attn_tokens = self.global_blocks[global_idx](attn_tokens, None)
            global_idx += 1
            # We force self.aa_block_size=1 now, so we don't reconstruct per-frame layout for
            # "intermediates" here. Consumers expect per-frame shapes only at the end of the
            # global stage, and skipping the intermediate unpack avoids extra views/concats.
            # NOTE: If aa_block_size > 1 in the future, reconstruct before appending to keep
            # shapes consistent with frame_intermediates: [B, S, P, C].

        if global_attn_mode == "all":
            tokens = attn_tokens.view(B, S, P, C)
        elif global_attn_mode == "registers":
            num_registers = ps - 1
            attended_registers_per_frame = attn_tokens.view(B, S, num_registers, C)
            special_per_frame = torch.cat([camera_toks_per_frame, attended_registers_per_frame], dim=2)
            special = special_per_frame.reshape(B, S * ps, C)
            tokens = torch.cat([special, patches], dim=1)
        elif global_attn_mode == "specials":
            tokens = torch.cat([attn_tokens, patches], dim=1)
        else:
            tokens = attn_tokens

        if global_attn_mode != "all":
            special = tokens[:, : S * ps, :].view(B, S, ps, C)
            patches = tokens[:, S * ps :, :].view(B, S, P - ps, C)
            tokens = torch.cat([special, patches], dim=2)
        intermediates.append(tokens)
        assert tokens.shape[-2] == P

        return tokens, global_idx, intermediates



def slice_expand_and_flatten(token_tensor: torch.Tensor, B: int, S: int) -> torch.Tensor:
    """
    Prepare specialized tokens of shape (1, 2, X, C) for multi-frame inputs.
    - Use index 0 for the first frame, index 1 for all remaining frames.
    - Expand along batch, concatenate along frames to (B, S, X, C), then
      flatten to (B*S, X, C).
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)
    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
