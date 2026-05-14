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
