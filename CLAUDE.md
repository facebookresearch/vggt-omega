# CLAUDE.md

This repository is a research-oriented release for VGGT-Omega. Keep the code
simple, readable, and easy for researchers to modify.

## General Rules

- Prefer straightforward implementations over framework-like abstractions.
- Keep APIs small and explicit. A user should be able to understand the main
  inference path by reading a few files.
- Prefer public-VGGT-style model entry points: the top-level model class should
  show which modules are connected, not read like a full experiment config.
- Avoid over-engineering for production safety, service deployment,
  multi-tenant use, or large configuration systems.
- Use simple Python configs, dataclasses, or small dictionaries when
  configuration is needed. Do not introduce complex config frameworks unless
  there is a clear, immediate need.
- Do not add training, fine-tuning, or benchmark reproduction code unless the
  release scope explicitly changes.
- Do not add broad infrastructure such as plugin systems, registries, dynamic
  dependency loaders, telemetry, or distributed job orchestration.
- Keep dependencies minimal and common in the computer-vision research
  ecosystem. Avoid adding a dependency for small helper logic.
- Optimize for a clean inference workflow: install, load checkpoint, load
  images, predict cameras and depth, visualize or export.
- Prefer readable tensor code with clear shapes over clever compact code.
- Add comments only where they clarify non-obvious model or geometry behavior.
- Match the existing VGGT release style where useful, but simplify it for the
  VGGT-Omega release scope.

## Model and API Structure

- Keep `VGGTOmega` thin. It should mostly assemble the patch embedder,
  aggregator, heads, and optional alignment head.
- Put natural architecture defaults in the module that owns them. For example,
  shared VGGT-Omega defaults for the aggregator, camera head, depth head, and
  alignment head should live in those class constructors, not in a large
  `_build_*` block inside `VGGTOmega`.
- Avoid private builder functions when the same clarity can be achieved with
  direct module construction and good defaults.
- Expose a small public constructor with familiar arguments such as
  `patch_size`, `embed_dim`, and the feature toggles we actually release.
- Put simple constructor defaults directly in the signature. Do not add
  `DEFAULT_*` constants or duplicate `self.*` attributes unless another part of
  the code actually needs to read them.
- Do not move one-file architecture values into module-level uppercase
  constants just to make the file look organized. Prefer the public VGGT style:
  keep defaults in the constructor signature or next to the module construction
  that uses them.
- When renaming modules or attributes that can change `state_dict` keys, record
  the old and new key prefixes in `docs/checkpoint_key_renames.md` in the same
  change.
- Do not expose switches for unreleased capabilities such as point, track,
  training, or fine-tuning.
- Keep preprocessing defaults, such as checkpoint-specific image size, outside
  the `nn.Module` unless the forward pass or module construction actually needs
  them.
- If a change is intended to be a cleanup or reorganization only, verify it
  against dirty code before considering it done.

## Release Checkpoints

- Plan for two public 1B checkpoints:
  - a 512-resolution reconstruction checkpoint for camera and depth inference
  - a 256-resolution checkpoint with language alignment
- Keep checkpoint selection simple, for example with a small named preset or a
  direct checkpoint path. Do not introduce a large registry or config system.
- The 512-resolution checkpoint should be the default for reconstruction
  examples unless a language-alignment example specifically needs the 256 model.
- Released checkpoints must be self-contained for inference. Model construction
  and checkpoint loading must not require downloading or separately caching
  DINOv3 pretrained weights.
- Keep the model class as a plain `torch.nn.Module`. Do not inherit
  `PyTorchModelHubMixin` or require `huggingface_hub` just to construct, load,
  or run the model.
- Keep checkpoint download helpers separate from model initialization. If a
  Hugging Face-specific dependency or compatibility layer seems necessary,
  discuss it first.

## Import Hygiene

- Be careful with package names. The copied dirty training code still uses the
  `vggt.*` namespace, and some development environments may also have unrelated
  or older `vggt` packages on `PYTHONPATH`.
- Before debugging model behavior, verify the active package path with:
  `python -c "import vggt; print(vggt.__file__)"`.
- Tests and examples should run from this repository or otherwise make the
  intended import path explicit. Do not assume `import vggt` points to the
  local release package unless it has been checked.
- Avoid adding more ambiguous top-level package names. Public user-facing APIs
  should prefer the `vggt_omega` package when we add the cleaned release wrapper.
- Do not keep compatibility aliases by default. Once a public API name is
  chosen, use that name consistently and remove the old names. If a change seems
  to require a compatibility alias or migration path, discuss it first instead
  of adding the alias silently.

## Naming

- Use `VGGT-Omega` as the public model name in papers, README text, release
  notes, checkpoint names, and model cards.
- Use `VGGTOmega` as the Python model class name.
- Use `vggt_omega` as the Python package name.
- Do not use `VGGTOMEGA` as a class name or compatibility alias.

## Precision Policy

- Default inference should run the backbone and aggregator under AMP with
  bfloat16 on CUDA when supported, falling back to float16 otherwise.
- Heads should run in float32 by disabling autocast around camera/depth/alignment
  heads. This matches the training setup for the z028 checkpoint: global AMP is
  enabled with bfloat16 by default, while the heads are kept in fp32.
- Keep this policy explicit in the model forward path rather than hiding it
  behind environment variables.
- The release inference path assumes CUDA. Do not add per-call CPU fallback
  wrappers such as `contextlib.nullcontext()` branches around autocast.

## Attention Backend Policy

- Default release code should work with PyTorch scaled dot product attention.
  On modern PyTorch/CUDA this uses the Flash Attention v2 backend when
  available, without requiring the user to install a separate flash-attn package.
- Optional Flash Attention v3 support can be documented as an advanced H100 path;
  on H100 it can be about 2x faster, but it must not be required for the default
  installation or quick start.
- If an optional Flash Attention v3 path is exposed, provide a simple fallback to
  PyTorch SDPA because flash-attn builds are hardware- and environment-sensitive.

## Code Style

- Keep files focused and reasonably short.
- Prefer simple defaults and direct calls over config dumps, registries, or
  indirection layers.
- Use descriptive names for tensors, especially for cameras, depth, masks,
  tokens, and image dimensions.
- Make defaults work for the common case. Expose only the few parameters users
  are likely to change.
- Fail with direct, helpful errors for missing checkpoints, invalid image paths,
  or unsupported tensor shapes.
- Do not hide important behavior behind environment variables or implicit
  global state.
- Preserve camera and geometry convention notes near the code that uses them.
- For behavior-preserving refactors, compare release outputs against the dirty
  training-code model on fixed inputs. Bitwise equality is preferred when the
  precision policy and inputs are identical.
- Do not run expensive checkpoint forward comparisons by default for small
  cleanup edits. Use compile/static checks first, and run checkpoint comparisons
  when explicitly requested or when the change could plausibly affect numerical
  behavior.

## DINOv3-Derived Code

- Treat files under `vggt_omega/models/layers` as DINOv3-derived building
  blocks. Keep them as close as practical to the public DINOv3 implementation.
- Revert training-time convenience edits unless they are required for released
  VGGT-Omega checkpoints or inference behavior.
- When a DINOv3-derived file needs an Omega-specific change, keep the change
  minimal and add a short `VGGT-Omega change:` comment explaining why it exists.
- Prefer passing Omega-specific settings from `VGGTOmega` or `Aggregator`
  instead of changing DINOv3 component defaults.

## Documentation Style

- Write README and docs for researchers who want to run the model quickly.
- Prefer minimal runnable examples over long explanations.
- Clearly state what is included and what is intentionally not included.
- Do not document training or evaluation workflows unless they are actually
  released in this repository.
