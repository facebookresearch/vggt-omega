# How We Built `VGGLinearHead`

This note records how we iteratively evolved `projects/vggt/models/heads/linear_variants/vgglinearhead.py` from a single-layer linear head to a configurable multi-variant head.

## 1) Starting Point

Initial implementation was a simplified head:

- uses only the last layer tokens
- one shared `LayerNorm`
- three projection heads (`proj`, `proj_conf`, `proj_mask`)
- pixel shuffle to full resolution
- always predicts mask

## 2) Multi-Scale Feature Path Added

We added an optional multi-scale path (kept backward-compatible by default):

- `intermediate_layer_idx` enables multi-scale mode
- each selected level:
  - token norm
  - reshape to 2D map
  - `projects` (1x1 conv projection)
  - `resize_layers` (configurable upsample variant)
- fused feature is then decoded by `proj / proj_conf / proj_mask`

Behavior:

- if `intermediate_layer_idx` is `None`, legacy single-layer path is used
- if provided, multi-scale path is used

## 3) Resize Variants (x4 Unification)

We made upsample behavior configurable:

- `multiscale_resize_type:`
  - `deconv`
  - `bilinear`
  - `bilinear_dw`
  - `bilinear_normal_conv`

Implementation was moved to:

- `projects/vggt/models/heads/linear_variants/head_utils.py`
  - `build_multiscale_resize_layer(...)`

## 4) Merge Variants

Added multi-scale merge mode:

- `multiscale_merge_type:`
  - `concat` (default)
  - `add_inplace`

For `add_inplace`, all `multiscale_out_channels` must be equal.

## 5) Norm Variants

Added:

- `multiscale_norm_type:`
  - `shared` (default)
  - `per_level`

Important DDP fix:

- when `use_multiscale=True` and `multiscale_norm_type="per_level"`, `self.norm` is set to `nn.Identity()` to avoid unused-parameter issues in DDP.

## 6) Mask Prediction Became Optional

Added:

- `predict_mask: bool = True`

If disabled:

- `proj_mask` is not built
- forward returns `mask=None`

## 7) Positional Embedding (RoPE-like) for Fused/Single Path

Added generic (not multiscale-only) positional options:

- `pos_embed_type: "none" | "rope2d" | "rope4d"`
- `pos_embed_base: float` (default `100.0`)
- `pos_embed_scale: float` (default `0.1`)

Applied on:

- fused feature map in multi-scale path
- token map (2D reshaped) in single-layer path

RoPE-like helper implemented in:

- `head_utils.py`
  - `build_rope_like_pos_embed(...)`

Note:

- legacy keys `multiscale_pos_embed_*` are intentionally rejected now
- use only `pos_embed_*`

## 8) Post Smoothing Conv

Added output smoothing option:

- `post_conv_type: "none" | "dw_conv" | "conv"`

Final decision:

- `post_conv` is applied **before** activation (`activate_head`) for more correct behavior.

## 9) Multi-Scale Head Internal Conv Variant

Added configurable optional spatial conv inside `_build_multiscale_head`:

- `multiscale_head_conv_type:`
  - `none` (default)
  - `dwconv`
  - `conv`

This affects both `proj_type="linear"` and `proj_type="mlp_self"` in multiscale mode.

## 10) Refactor / Cleanups

- extracted resize + rope-like utility code into `head_utils.py`
- replaced Identity sentinel with `None` sentinel in multiscale head builder
- tightened parameter validation and error messages

## 11) Test Coverage

Main test file:

- `projects/vggt/e2e_test/test_vgglinearhead_multiscale.py`

Covers:

- legacy path compatibility
- multiscale path shape/dtype correctness
- all resize variants
- merge variants (`concat`, `add_inplace`)
- norm variants (`shared`, `per_level`)
- optional mask path
- pos embed variants and scale behavior
- post-conv variants and ordering (before activation)
- multiscale head conv variants
- invalid config handling

## 12) Current Key Config Surface

Core toggles:

- `intermediate_layer_idx`
- `multiscale_out_channels`
- `multiscale_upsample_factor`
- `multiscale_resize_type`
- `multiscale_merge_type`
- `multiscale_norm_type`
- `multiscale_head_conv_type`
- `predict_mask`
- `pos_embed_type`
- `pos_embed_base`
- `pos_embed_scale`
- `post_conv_type`

This gives one head implementation that supports many experimental variants while keeping defaults backward-compatible in behavior.

