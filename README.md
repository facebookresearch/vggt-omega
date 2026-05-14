<div align="center">
<h1>VGGT-&Omega;</h1>

<a href="http://vggt-omega.github.io/" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page">
</a>

<p>
  <span class="author"><a href="https://jytime.github.io/">Jianyuan Wang</a><sup>1,2</sup></span>
  <span class="author"><a href="https://silent-chen.github.io/">Minghao Chen</a><sup>1</sup></span>
  <span class="author"><a href="https://scholar.google.com/citations?user=FUDsZkEAAAAJ&amp;hl=zh-CN">Shangzhan Zhang</a><sup>1</sup></span>
  <span class="author"><a href="https://nikitakaraevv.github.io/">Nikita Karaev</a><sup>1</sup></span>
  <br>
  <span class="author"><a href="https://demuc.de/">Johannes Schönberger</a><sup>2</sup></span>
  <span class="author"><a href="https://scholar.google.com/citations?user=IJidh-UAAAAJ&amp;hl=fr">Patrick Labatut</a><sup>2</sup></span>
  <span class="author"><a href="https://scholar.google.com/citations?user=lJ_oh2EAAAAJ&amp;hl=en">Piotr Bojanowski</a><sup>2</sup></span>
  <span class="author"><a href="https://d-novotny.github.io/">David Novotny</a></span>
  <span class="author"><a href="https://www.robots.ox.ac.uk/~vedaldi/">Andrea Vedaldi</a><sup>1,2</sup></span>
  <span class="author"><a href="https://chrirupp.github.io/">Christian Rupprecht</a><sup>1</sup></span>
</p>

**<sup>1</sup>[Visual Geometry Group, University of Oxford](https://www.robots.ox.ac.uk/~vgg/)**; **<sup>2</sup>[Meta AI](https://ai.facebook.com/research/)**
</div>

```bibtex
@inproceedings{wang2026vggtomega,
  title={VGGT-{$\Omega$}},
  author={Wang, Jianyuan and Chen, Minghao and Zhang, Shangzhan and Karaev, Nikita and Sch{\"o}nberger, Johannes and Labatut, Patrick and Bojanowski, Piotr and Novotny, David and Vedaldi, Andrea and Rupprecht, Christian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

## Model Zoo

| Model | Resolution | Outputs | Access |
| --- | --- | --- | --- |
| `VGGT-Omega-1B-512` | 512 | camera, depth, camera/register tokens | [Request access](https://huggingface.co/facebook/VGGT-Omega-1B-512) |
| `VGGT-Omega-1B-256-Text-Alignment` | 256 | camera, depth, camera/register tokens, text alignment | [Request access](https://huggingface.co/facebook/VGGT-Omega-1B-256-Text-Alignment) |

VGGT-&Omega; checkpoint access requires review. After approval, place the
checkpoint file on your machine and pass the local path to the examples or
demo. The checkpoints are raw PyTorch `state_dict` files and load directly with
`model.load_state_dict(torch.load(...))`.

## Quick Start

First, clone this repository and install the dependencies:

```bash
git clone git@github.com:facebookresearch/vggt-omega.git
cd vggt-omega
pip install -r requirements.txt
pip install -e .
```

Alternatively, you can install VGGT-&Omega; as a package
(<a href="docs/package.md">click here</a> for details).

Now, try the model with a few lines of code:

```python
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import pose_encoding_to_extri_intri

device = "cuda"
checkpoint_path = "checkpoints/VGGT-Omega-1B-512/model.pt"
image_names = [
    "examples/znz_20260430_6_crop_top100_00.png",
    "examples/znz_20260430_6_crop_top100_01.png",
    "examples/znz_20260430_6_crop_top100_02.png",
]

model = VGGTOmega().to(device).eval()
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

images = load_and_preprocess_images(image_names, image_resolution=512).to(device)

with torch.inference_mode():
    predictions = model(images)

extrinsics, intrinsics = pose_encoding_to_extri_intri(
    predictions["pose_enc"],
    predictions["images"].shape[-2:],
)

depth = predictions["depth"]
depth_conf = predictions["depth_conf"]
camera_and_register_tokens = predictions["camera_and_register_tokens"]
camera_tokens = camera_and_register_tokens[:, :, :1]
registers = camera_and_register_tokens[:, :, 1:]
```

VGGT-&Omega; does not require a separate DINOv3 pretrained-weight download.
The DINOv3-derived backbone weights are part of the released checkpoints.

## Detailed Usage

<details>
<summary>Click to expand</summary>

### Outputs

For input images with shape `[S, 3, H, W]`, `VGGTOmega.forward()` adds a batch
dimension and returns tensors with batch shape `[1, S, ...]`.

| Key | Shape | Description |
| --- | --- | --- |
| `pose_enc` | `[B, S, 9]` | Camera encoding: translation, quaternion, vertical FoV, horizontal FoV. |
| `depth` | `[B, S, H, W, 1]` | Predicted depth map. |
| `depth_conf` | `[B, S, H, W]` | Depth confidence. |
| `camera_and_register_tokens` | `[B, S, 17, 2048]` | Final-layer camera token followed by 16 registers / scene tokens. |
| `images` | `[B, S, 3, H, W]` | Preprocessed input images, returned in eval mode. |

The first token in `camera_and_register_tokens` is the camera token. The
remaining 16 tokens are registers, also called scene tokens in the paper.

</details>

## Interactive Demo

Install the demo dependencies:

```bash
pip install -r requirements_demo.txt
```

Launch the Gradio demo with a local checkpoint path:

```bash
python demo_gradio.py \
  --checkpoint checkpoints/VGGT-Omega-1B-512/model.pt \
  --image-resolution 512
```

The demo accepts uploaded images or a video, runs camera and depth inference,
and visualizes the depth-unprojected point cloud and predicted cameras as a GLB
scene.

## License

See the [LICENSE](./LICENSE) file for details about the license under which
this code is made available.

[^release]: This Release is intended to support the open source research community.

