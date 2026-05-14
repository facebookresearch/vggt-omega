# Alternative Installation Methods

This document explains how to install VGGT-&Omega; as a package using different
package managers.

## Prerequisites

Before installing VGGT-&Omega;, install PyTorch and torchvision for your CUDA
environment. For example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Installation Options

### Install with pip

```bash
pip install -e .
```

### Install and run with uv

[uv](https://docs.astral.sh/uv/) is a fast Python package installer and
resolver.

```bash
uv run --extra demo demo_gradio.py \
  --checkpoint checkpoints/VGGT-Omega-1B-512/model.pt \
  --image-resolution 512
```
