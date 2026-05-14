# VGGT-Omega Release Progress

Last updated: 2026-05-14

This document tracks implementation and cleanup progress for the inference-only
VGGT-Omega release. It is intended as a lightweight engineering note, not as
user-facing documentation.

## Current Release Scope

The first release targets a minimal research-oriented inference package:

- Install the package.
- Load released checkpoints.
- Run camera and depth inference.
- Visualize or export predictions.

Training, fine-tuning, evaluation, and reproduction scripts are out of scope for
this release.

## Model Naming

- Public project/model name: `VGGT-Omega`.
- Python module class name: `VGGTOmega`.
- Model entry implementation: `vggt_omega.models.vggt_omega`.
- No compatibility aliases are kept. If an alias or compatibility path seems
  necessary later, discuss it before adding it.

## Package Layout

The package follows the public VGGT layout where practical:

- `vggt_omega.models` contains the model entry point and aggregator.
- `vggt_omega.models.layers` contains the DINOv3-derived vision components.
- `vggt_omega.models.heads` contains camera and depth heads.
- `vggt_omega.utils` contains camera, geometry, pose encoding, rotation, and
  image-loading utilities.

## Layers Cleanup: Pass 1

Status: complete for the first pass.

The `vggt_omega.models.layers` package is treated as the DINOv3-derived vision
component. The cleanup goal is to keep these files as close as practical to
public DINOv3 code while preserving only the changes required by the
VGGT-Omega checkpoint.

Files currently matching public DINOv3 exactly:

- `layer_scale.py`
- `patch_embed.py`
- `rms_norm.py`
- `rope_position_encoding.py`
- `utils.py` matches DINOv3's public `utils.py`

Files with intentional differences:

- `attention.py`
  - Uses local imports instead of `dinov3.*` imports.
  - Adds `use_qk_norm`, `q_norm`, and `k_norm` support because the aggregator
    checkpoint was trained with Q/K normalization.
- `block.py`
  - Uses local imports instead of `dinov3.*` imports.
  - Passes `use_qk_norm` through to `SelfAttention`.
  - Removes public DINOv3's import-time `torch._dynamo.config` mutations. The
    VGGT-Omega training code had those lines disabled, and the release does not
    provide a `torch.compile` path by default.
- `ffn_layers.py`
  - Uses local imports instead of `dinov3.*` imports.
- `vision_transformer.py`
  - Lives under `layers/` so the DINOv3-derived vision transformer stays with
    its supporting layer components.
  - Uses local imports through `layers/__init__.py`, following the public
    DINOv3 import style.
  - Defaults `forward(..., is_training=True)` so it returns feature dictionaries
    consumed by the VGGT-Omega aggregator.
- `__init__.py`
  - Exports only the layer objects included in this repo.
  - Does not export DINOv3 components that were not copied, such as FP8 or
    sparse linear helpers.

Files intentionally not included from public DINOv3 `layers/`:

- `dino_head.py`
- `fp8_linear.py`
- `sparse_linear.py`

Current code does not reference these files, and they are not needed for
VGGT-Omega inference.

## RoPE Behavior

Public DINOv3 components default to `normalize_coords="separate"`. VGGT-Omega's
released checkpoint was trained with max-normalized RoPE coordinates.

The release keeps the DINOv3-derived class defaults close to upstream, but
`VGGTOmega` explicitly constructs:

- patch embed RoPE with `pos_embed_rope_normalize_coords="max"`
- aggregator RoPE with `rope_normalize_coords="max"`

`VGGTOmega` now warns if either constructed RoPE module is not using `"max"`.

## Camera FoV Activation

The camera head keeps the training-time behavior for FoV prediction:

- Apply the configured FoV activation, currently ReLU by default.
- Add `0.01` to the FoV channels for numerical stability.

This makes the predicted FoV strictly positive before converting FoV back to
focal length.

## Models Cleanup: Pass 1

Status: complete for the first pass.

The first cleanup pass outside `layers/` removed code paths that only served
training, config compatibility, or unused head variants:

- Removed gradient checkpointing branches from the aggregator, camera head, and
  dense head.
- Removed the dense-head checkpoint wrapper.
- Removed the aggregator's Pi3/training initialization mode.
- Removed patch-embed intermediate outputs from the release model path.
- Removed camera-head training initialization helpers.
- Removed unused dense-head feature-only, no-confidence, half-dim, frame-chunk, and
  mask-prediction branches.
- Updated head autocast disabling to `torch.autocast(device_type="cuda",
  enabled=False)`.

The pass intentionally kept checkpoint-architecture choices such as
`global_attn_mode`, `global_attn_indices`, `use_dino_clsreg`, and RoPE-related
options until both release checkpoints could be checked. These were simplified
later in `Aggregator Cleanup: Pass 1`.

## Model Entry Cleanup: Pass 1

Status: complete for the first pass.

`vggt_omega.models.vggt_omega` now follows the public VGGT reading style more
closely: the `VGGTOmega` class appears near the top of the file, and its
constructor only wires together the aggregator, camera head, depth head, and
optional alignment head.

The shared VGGT-Omega architecture defaults now live in the corresponding
release components:

- `Aggregator`
- `CameraHead`
- `DenseHead`
- `TextAlignmentHead`

This lets `VGGTOmega` instantiate those modules with simple public-style
arguments such as `patch_size`, `embed_dim`, and feature toggles, instead of
carrying checkpoint-specific `_build_aggregator()` or
`_build_dense_head()` helper functions in the model entry file.

The file still keeps release-specific helpers for checkpoint loading,
backbone/aggregator autocast, head fp32 execution, and RoPE behavior warnings.
These replace the public VGGT `PyTorchModelHubMixin` path and preserve the
training-time precision behavior.

## Aggregator Cleanup: Pass 1

Status: complete for the first simplified release shape.

The aggregator now exposes only the architecture choices used by the released
checkpoints. Training-time and exploration-only switches were removed from the
public code path:

- Removed dynamic block/FFN/dtype registries.
- Removed unused global-attention modes and partial-ratio scheduling.
- Removed DINO cls/register-token merging.
- Removed patch-token residual and patch-embed RoPE matching branches.
- Removed global RoPE, gradient checkpointing leftovers, and custom ViT init
  helpers.

The release aggregator keeps the trained behavior fixed: frame/global
alternating attention, camera/register special tokens, max-normalized RoPE on
frame patch tokens, Q/K normalization, and special-only global attention at
blocks `[2, 6, 9, 14, 20]`.

No checkpoint key rename was needed for this pass.

## Patch Embed Ownership: Pass 1

Status: complete.

The DINOv3 patch embedder now lives inside `Aggregator`. The top-level
`VGGTOmega` model no longer stores `self.patch_embed` or passes a patch-embed
module into `Aggregator.forward()`.

This makes the release inference path read as:

- `VGGTOmega`: input handling, precision policy, and heads.
- `Aggregator`: image normalization, DINOv3 patch embedding, special tokens,
  RoPE, and alternating attention.

This pass changed checkpoint key ownership:

- `patch_embed.` -> `aggregator.patch_embed.`

The mapping is recorded in `docs/checkpoint_key_renames.md` and applied by
`VGGTOmega.from_checkpoint()`.

## Head Cleanup: Pass 2

Status: complete for the first simplified release shape.

The camera, depth, and text-alignment heads now expose only the small set of
release-facing constructor arguments needed by `VGGTOmega`. Training and
exploration switches were removed from the public code path, while module names
with checkpoint weights were preserved.

- `CameraHead` now returns `pose_enc` directly instead of a one-element
  `pose_enc_list`.
- `DenseHead` now hardcodes the released depth/confidence behavior:
  positional embedding on, linear prediction projections, depth `exp`, and
  confidence `1 + exp`.
- `TextAlignmentHead` now contains only the released student branch.
- `vggt_omega.models.heads.head_act` was removed because the remaining head
  activations are fixed and local.

## Head Naming: Pass 1

Status: complete for code names.

The release head names now describe the public model structure directly:

- `camera_head_linear.py` -> `camera_head.py`
- `CameraHeadLinear` -> `CameraHead`
- `dpt_linear_head.py` -> `dense_head.py`
- `DPTLinearHead` -> `DenseHead`

The top-level model attribute for dense prediction is now `dense_head` instead
of `depth_head`. This changes checkpoint keys from `depth_head.*` to
`dense_head.*`; the mapping is recorded in `docs/checkpoint_key_renames.md`.

## Head Trunk Naming: Pass 1

Status: complete for code names.

The head-local self-attention blocks are now named `trunk`, matching public
VGGT's style for the main processing stack inside a head. The old
`extra_attention` name was a research-time name and made the blocks look like a
temporary add-on.

The relevant checkpoint key mappings are recorded in
`docs/checkpoint_key_renames.md`.

`VGGTOmega.from_checkpoint()` now applies the current key rename rules through
`rename_state_dict_keys(state_dict, rules)` before strict loading. Future
renames should extend the rule list and the record together.

## Text Alignment Checkpoint: Pass 1

Status: complete for the first pass.

The 256-resolution text-aligned checkpoint is now handled as a student-only
release checkpoint:

- Source checkpoint:
  `/home/jianyuan/ckpts/round_final/w008_linear_256_text_align_e30.pt`
- Clean checkpoint:
  `/home/jianyuan/ckpts/round_final/w008_linear_256_text_align_e30_clean.pt`
- Removed training state (`optimizer`, `scaler`, etc.).
- Removed all `alignment_head.teacher.*` Qwen/VLM parameters.
- Kept `alignment_head.student.*` parameters for release inference.

`VGGTOmega.from_checkpoint()` detects `alignment_head.student.*` keys and
constructs the student-only alignment head automatically. Checkpoint metadata
can still record preprocessing information such as image size, but the
`nn.Module` does not store that metadata unless it is needed for forward or
module construction. The release code does not instantiate Qwen and does not
depend on `transformers`.

Verification on the 8 example frames resized to 256 matched dirty code exactly
when dirty code was run with the same training-time precision policy
(backbone/aggregator AMP, heads/alignment fp32):

- `pose_enc`: max abs diff 0
- `depth`: max abs diff 0
- `depth_conf`: max abs diff 0
- `alignment_student_embedding`: max abs diff 0
- `alignment_student_token`: max abs diff 0

## Verification Notes

The current release model path has been checked against the training-code model
on a fixed input:

- Checkpoint loading is strict with no missing or unexpected keys.
- Camera pose encoding, depth, and depth confidence matched exactly in the
  comparison run.
- After moving the Omega architecture values into module defaults, both release
  checkpoints still matched dirty code exactly:
  - 512 no-alignment checkpoint: `pose_enc`, `depth`, `depth_conf`
  - 256 text-alignment checkpoint: `pose_enc`, `depth`, `depth_conf`,
    `alignment_student_embedding`, `alignment_student_token`
- After simplifying the aggregator, both release checkpoints still strict-loaded
  and matched dirty code exactly on the same fixed inputs:
  - 512 no-alignment checkpoint: max abs diff 0 for `pose_enc`, `depth`,
    `depth_conf`
  - 256 text-alignment checkpoint: max abs diff 0 for `pose_enc`, `depth`,
    `depth_conf`, `alignment_student_embedding`, `alignment_student_token`
- After moving the patch embedder into `Aggregator`, both release checkpoints
  still strict-loaded and matched dirty code exactly on the same fixed inputs:
  - 512 no-alignment checkpoint: max abs diff 0 for `pose_enc`, `depth`,
    `depth_conf`
  - 256 text-alignment checkpoint: max abs diff 0 for `pose_enc`, `depth`,
    `depth_conf`, `alignment_student_embedding`, `alignment_student_token`

Eight example frames have been extracted into `examples/` for future
training-vs-release comparison tests.

## Open Cleanup Items

- Continue simplifying files outside `layers/`, especially heads and utilities,
  while preserving exact checkpoint behavior.
