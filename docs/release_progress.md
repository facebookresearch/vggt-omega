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
  depth head.
- Removed the DPT checkpoint wrapper.
- Removed the aggregator's Pi3/training initialization mode.
- Removed patch-embed intermediate outputs from the release model path.
- Removed camera-head training initialization helpers.
- Removed unused DPT feature-only, no-confidence, half-dim, frame-chunk, and
  mask-prediction branches.
- Updated head autocast disabling to `torch.autocast(device_type="cuda",
  enabled=False)`.

The pass intentionally kept checkpoint-architecture choices such as
`global_attn_mode`, `global_attn_indices`, `use_dino_clsreg`, `proj_type`, and
RoPE-related options. These affect possible checkpoint layouts or computation
paths, so they should only be removed after checking all checkpoints intended
for release.

## Model Entry Cleanup: Pass 1

Status: complete for the first pass.

`vggt_omega.models.vggt_omega` now follows the public VGGT reading style more
closely: the `VGGTOmega` class appears near the top of the file, and its
constructor only wires together the patch embedder, aggregator, camera head,
depth head, and optional alignment head.

The shared VGGT-Omega architecture defaults now live in the corresponding
release components:

- `Aggregator`
- `CameraHeadLinear`
- `DPTLinearHead`
- `TextAlignmentHead`

This lets `VGGTOmega` instantiate those modules with simple public-style
arguments such as `img_size`, `patch_size`, `embed_dim`, and feature toggles,
instead of carrying checkpoint-specific `_build_aggregator()` or
`_build_depth_head()` helper functions in the model entry file.

The file still keeps release-specific helpers for checkpoint loading,
backbone/aggregator autocast, head fp32 execution, and RoPE behavior warnings.
These replace the public VGGT `PyTorchModelHubMixin` path and preserve the
training-time precision behavior.

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
constructs the student-only alignment head automatically. It also reads the
clean checkpoint metadata so the text-aligned checkpoint records
`img_size=256`, while the default no-alignment checkpoint remains `img_size=512`.
The release code does not instantiate Qwen and does not depend on
`transformers`.

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

Eight example frames have been extracted into `examples/` for future
training-vs-release comparison tests.

## Open Cleanup Items

- Continue simplifying files outside `layers/`, especially heads and utilities,
  while preserving exact checkpoint behavior.
