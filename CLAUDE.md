# CLAUDE.md

This repository is a research-oriented release for VGGT-Omega. Keep the code
simple, readable, and easy for researchers to modify.

## General Rules

- Prefer straightforward implementations over framework-like abstractions.
- Keep APIs small and explicit. A user should be able to understand the main
  inference path by reading a few files.
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

## Precision Policy

- Default inference should run the backbone and aggregator under AMP with
  bfloat16 on CUDA when supported, falling back to float16 otherwise.
- Heads should run in float32 by disabling autocast around camera/depth heads.
  This matches the training setup for the z028 checkpoint: global AMP is enabled
  with bfloat16 by default, while `enable_head_amp=False` and head-level
  `disable_last_layer_amp=True` keep the heads in fp32.
- Keep this policy explicit in the model forward path rather than hiding it
  behind environment variables.

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
- Use descriptive names for tensors, especially for cameras, depth, masks,
  tokens, and image dimensions.
- Make defaults work for the common case. Expose only the few parameters users
  are likely to change.
- Fail with direct, helpful errors for missing checkpoints, invalid image paths,
  or unsupported tensor shapes.
- Do not hide important behavior behind environment variables or implicit
  global state.
- Preserve camera and geometry convention notes near the code that uses them.

## Documentation Style

- Write README and docs for researchers who want to run the model quickly.
- Prefer minimal runnable examples over long explanations.
- Clearly state what is included and what is intentionally not included.
- Do not document training or evaluation workflows unless they are actually
  released in this repository.
