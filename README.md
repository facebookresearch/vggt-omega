<div align="center">
<h1>VGGT-Omega</h1>

<a href="http://vggt-omega.github.io/">
  <img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page">
</a>
<a href="https://huggingface.co/facebook/VGGT-Omega-1B-512">
  <img src="https://img.shields.io/badge/Access-1B--512-blue" alt="512 Checkpoint Access">
</a>
<a href="https://huggingface.co/facebook/VGGT-Omega-1B-256-Text-Alignment">
  <img src="https://img.shields.io/badge/Access-1B--256--Text--Alignment-blue" alt="256 Text Alignment Checkpoint Access">
</a>

**[Visual Geometry Group, University of Oxford](https://www.robots.ox.ac.uk/~vgg/)**; **[Meta AI](https://ai.facebook.com/research/)**

Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes Schonberger,
Patrick Labatut, Piotr Bojanowski, David Novotny, Andrea Vedaldi, Christian Rupprecht
</div>

```bibtex
@inproceedings{wang2026vggtomega,
  title={VGGT-{$\Omega$}},
  author={Wang, Jianyuan and Chen, Minghao and Zhang, Shangzhan and Karaev, Nikita and Schonberger, Johannes and Labatut, Patrick and Bojanowski, Piotr and Novotny, David and Vedaldi, Andrea and Rupprecht, Christian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

## Overview

VGGT-Omega is a feed-forward reconstruction model for static and dynamic
scenes. Given one or more input images, it predicts camera parameters and depth
maps in a single forward pass.[^release]

VGGT-Omega builds on VGGT with DINOv3 image features, register attention, a
lightweight camera head, and a single dense depth head. The model also exposes
its camera/register tokens: compact scene-level features that can be used for
downstream spatial understanding tasks.

## Model Zoo

| Model | Resolution | Outputs | Access |
| --- | --- | --- | --- |
| `VGGT-Omega-1B-512` | 512 | camera, depth, camera/register tokens | [Request access](https://huggingface.co/facebook/VGGT-Omega-1B-512) |
| `VGGT-Omega-1B-256-Text-Alignment` | 256 | camera, depth, camera/register tokens, text alignment | [Request access](https://huggingface.co/facebook/VGGT-Omega-1B-256-Text-Alignment) |

The checkpoints are raw PyTorch `state_dict` files and load directly with
`model.load_state_dict(torch.load(...))`. Checkpoint access requires review;
after approval, download the checkpoint yourself and pass its local path to the
examples or demo.

## Installation

Install PyTorch for your CUDA environment, then install VGGT-Omega:

```bash
git clone <repo-url>
cd vggt-omega
pip install -r requirements.txt
pip install -e .
```

VGGT-Omega does not require downloading separate DINOv3 pretrained weights.
The DINOv3-derived backbone weights are part of the released checkpoints.

## Quick Start

```python
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import pose_encoding_to_extri_intri

device = "cuda"
checkpoint_path = "checkpoints/VGGT-Omega-1B-512/model.pt"
image_paths = [
    "examples/znz_20260430_6_crop_top100_00.png",
    "examples/znz_20260430_6_crop_top100_01.png",
    "examples/znz_20260430_6_crop_top100_02.png",
]

model = VGGTOmega().to(device).eval()
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

images = load_and_preprocess_images(image_paths, image_resolution=512).to(device)

with torch.inference_mode():
    predictions = model(images)

extrinsics, intrinsics = pose_encoding_to_extri_intri(
    predictions["pose_enc"],
    predictions["images"].shape[-2:],
)

depth = predictions["depth"]
depth_conf = predictions["depth_conf"]
camera_and_register_tokens = predictions["camera_and_register_tokens"]
registers = camera_and_register_tokens[:, :, 1:]
```

For the language-aligned checkpoint:

```python
model = VGGTOmega(enable_alignment=True).to(device).eval()
model.load_state_dict(
    torch.load("checkpoints/VGGT-Omega-1B-256-Text-Alignment/model.pt", map_location="cpu")
)

images = load_and_preprocess_images(image_paths, image_resolution=256).to(device)

with torch.inference_mode():
    predictions = model(images)

text_embedding = predictions["text_alignment_embedding"]
```

## Outputs

For input images with shape `[S, 3, H, W]`, `VGGTOmega.forward()` adds a batch
dimension and returns tensors with batch shape `[1, S, ...]`.

| Key | Shape | Description |
| --- | --- | --- |
| `pose_enc` | `[B, S, 9]` | Camera encoding: translation, quaternion, vertical FoV, horizontal FoV. |
| `depth` | `[B, S, H, W, 1]` | Predicted depth map. |
| `depth_conf` | `[B, S, H, W]` | Depth confidence. |
| `camera_and_register_tokens` | `[B, S, 17, 2048]` | Final-layer camera token followed by 16 registers / scene tokens. |
| `images` | `[B, S, 3, H, W]` | Preprocessed input images, returned in eval mode. |
| `text_alignment_embedding` | `[B, 2048]` | Returned by the text-alignment checkpoint. |
| `text_alignment_token` | `[B, 2048]` | Returned by the text-alignment checkpoint. |

The first token in `camera_and_register_tokens` is the camera token. The
remaining 16 tokens are registers, also called scene tokens in the paper.

## Registers

VGGT-Omega uses register attention to exchange information across frames through
the register tokens. These registers form a compact representation of the scene
and can be reused as geometry-aware features for downstream tasks.

The text-alignment checkpoint adds a small readout head that attends to the
camera/register tokens and returns a normalized sequence embedding.

See [docs/registers.md](docs/registers.md) for details.

## Preprocessing

The default image loader uses `balanced` preprocessing. It keeps the total
number of patch tokens close to the requested resolution while allowing
non-square image shapes. Before resizing, extreme aspect ratios are
center-cropped into the range `[0.5, 2.0]`.

```python
images = load_and_preprocess_images(image_paths, image_resolution=512)
```

See [docs/preprocessing.md](docs/preprocessing.md) for the supported modes.

## Interactive Demo

Install the demo dependencies:

```bash
pip install -r requirements_demo.txt
```

Launch the Gradio demo:

```bash
python demo_gradio.py --checkpoint checkpoints/VGGT-Omega-1B-512/model.pt
```

The demo accepts uploaded images or a video, runs camera and depth inference,
and visualizes the depth-unprojected point cloud and predicted cameras as a GLB
scene.

## Runtime

VGGT-Omega uses PyTorch scaled dot product attention by default. On modern CUDA
setups this usually dispatches to the flash attention v2 backend. Flash
attention v3 can be about twice as fast on H100 GPUs in our testing, but it is
not required for the default setup.

The backbone and aggregator run under mixed precision: bfloat16 when supported,
otherwise float16. The camera, depth, and alignment heads run in float32.

## Documentation

- [Installation](docs/installation.md)
- [Checkpoints](docs/checkpoints.md)
- [Inference](docs/inference.md)
- [Image Preprocessing](docs/preprocessing.md)
- [Camera and Register Tokens](docs/registers.md)

## License

VGGT-Omega is released under the FAIR Noncommercial Research License. See
[LICENSE](LICENSE) for the full license text.

## Acknowledgements

VGGT-Omega builds on VGGT and uses DINOv3-derived vision transformer
components. We thank the broader 3D vision, reconstruction, and representation
learning communities for the many open research projects that made this work
possible.

[^release]: This Release is intended to support the open source research community.
