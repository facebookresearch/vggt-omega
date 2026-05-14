# VGGT-Omega

VGGT-Omega is a feed-forward reconstruction model for static and dynamic
scenes. Given one or more input images, it predicts camera parameters and depth
maps in a single forward pass, and exposes the learned camera/register tokens
for downstream spatial understanding tasks.[^release]

VGGT-Omega builds on VGGT with a simpler release architecture: DINOv3 image
features, alternating frame/inter-frame attention, register attention, a camera
head, and one dense depth head. This repository is an inference-only release.
It does not include training, fine-tuning, evaluation, or data annotation code.

Project page: <http://vggt-omega.github.io/>

## News

- Initial release scope: installation, checkpoints, camera/depth inference,
  camera/register token outputs, visualization, and export utilities.
- Two 1B checkpoints are planned: a 512-resolution reconstruction checkpoint
  and a 256-resolution checkpoint with a language-alignment head.

## Model Zoo

| Model | Resolution | Outputs | Notes |
| --- | --- | --- | --- |
| `VGGT-Omega-1B-512` | 512 | camera, depth, camera/register tokens | Main reconstruction checkpoint. |
| `VGGT-Omega-1B-256-Text-Alignment` | 256 | camera, depth, camera/register tokens, text alignment | Adds a lightweight language-alignment head. |

The released checkpoint files are raw PyTorch `state_dict` files and can be
loaded directly with `model.load_state_dict(torch.load(...))`.

See [docs/checkpoints.md](docs/checkpoints.md) for checkpoint details and
download commands.

## Installation

Install PyTorch for your CUDA environment first, then install this package:

```bash
git clone <repo-url>
cd vggt-omega
pip install -r requirements.txt
pip install -e .
```

VGGT-Omega does not require downloading separate DINOv3 pretrained weights.
The DINOv3-derived backbone weights are part of the released checkpoints.

More installation notes are in [docs/installation.md](docs/installation.md).

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

More examples are in [docs/inference.md](docs/inference.md).

## Outputs

For an input sequence with shape `[S, 3, H, W]`, `VGGTOmega.forward()` adds a
batch dimension and returns tensors with batch shape `[1, S, ...]`.

| Key | Shape | Description |
| --- | --- | --- |
| `pose_enc` | `[B, S, 9]` | Camera encoding: translation, quaternion, vertical FoV, horizontal FoV. |
| `depth` | `[B, S, H, W, 1]` | Predicted depth map. |
| `depth_conf` | `[B, S, H, W]` | Depth confidence. |
| `camera_and_register_tokens` | `[B, S, 17, 2048]` | Final-layer camera token followed by 16 registers / scene tokens. |
| `images` | `[B, S, 3, H, W]` | Preprocessed input images, returned in eval mode. |
| `text_alignment_embedding` | `[B, 2048]` | Only for the text-alignment checkpoint. |
| `text_alignment_token` | `[B, 2048]` | Only for the text-alignment checkpoint. |

The first token in `camera_and_register_tokens` is the camera token. The
remaining 16 tokens are registers, also called scene tokens in the paper.

## Preprocessing

The default preprocessing mode is `balanced`. It keeps the total number of
patch tokens close to the requested resolution while allowing non-square image
shapes. Before resizing, extreme aspect ratios are center-cropped into the
range `[0.5, 2.0]`.

```python
images = load_and_preprocess_images(image_paths, image_resolution=512)
```

See [docs/preprocessing.md](docs/preprocessing.md) for the supported modes.

## Runtime

VGGT-Omega uses PyTorch scaled dot product attention by default. On modern CUDA
setups this usually dispatches to the flash attention v2 backend. Flash
attention v3 can be about twice as fast on H100 GPUs in our testing, but it is
not required by this release.

The backbone and aggregator run under mixed precision: bfloat16 when supported,
otherwise float16. The camera, depth, and alignment heads run in float32,
matching the training-time precision policy.

## Visualization and Export

Visualization and export utilities will be kept lightweight and inference-only.
The first target is a simple camera/depth viewer and COLMAP-style export based
on the predicted cameras and depth maps.

## Scope

This repository includes inference code and documentation for released
checkpoints. It intentionally does not include:

- training or fine-tuning code
- benchmark evaluation or reproduction scripts
- the full data annotation pipeline
- Qwen/VLM teacher weights used during language-alignment training

## License

VGGT-Omega is released under the FAIR Noncommercial Research License. See
[LICENSE](LICENSE) for the full license text.

## Citation

```bibtex
@inproceedings{wang2026vggtomega,
  title={VGGT-{$\Omega$}},
  author={Wang, Jianyuan and Chen, Minghao and Zhang, Shangzhan and Karaev, Nikita and Schonberger, Johannes and Labatut, Patrick and Bojanowski, Piotr and Novotny, David and Vedaldi, Andrea and Rupprecht, Christian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

## Acknowledgements

VGGT-Omega builds on VGGT and uses DINOv3-derived vision transformer
components. We thank the broader 3D vision, reconstruction, and representation
learning communities for the many open research projects that made this work
possible.

[^release]: This Release is intended to support the open source research community.
