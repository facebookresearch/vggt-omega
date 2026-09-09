<div align="center">
<h1>VGGT-&Omega;</h1>

<a href="http://vggt-omega.github.io/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2605.15195" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/arXiv-2605.15195-b31b1b" alt="arXiv"></a>
<a href="https://huggingface.co/spaces/facebook/vggt-omega"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>

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
  <br>
  <span class="author"><a href="https://www.robots.ox.ac.uk/~vedaldi/">Andrea Vedaldi</a><sup>1,2</sup></span>
  <span class="author"><a href="https://chrirupp.github.io/">Christian Rupprecht</a><sup>1</sup></span>
</p>

**<sup>1</sup>[Visual Geometry Group, University of Oxford](https://www.robots.ox.ac.uk/~vgg/)**; **<sup>2</sup>[Meta AI](https://ai.facebook.com/research/)**
</div>

## Updates

- [Sep 8, 2026] We have released the [`training`](training) code and a [checkpoint from an additional training run](https://huggingface.co/facebook/VGGT-Omega/blob/main/vggt_omega_1b_416_reproduce.pt) to further validate reproducibility and address a potential concern with the original checkpoint. The new checkpoint should serve as the reference for future comparisons on the reported benchmarks.

## Pretrained models

Before using the models, please request access to the checkpoints [here](https://huggingface.co/facebook/VGGT-Omega). Once your request is approved, you can download the checkpoints. Please note that access requests are reviewed by an automated process based on the information provided in the request. The authors are not involved in the review process and cannot approve or reject individual applications. However, the [🤗 Hugging Face demo](https://huggingface.co/spaces/facebook/vggt-omega) is available to everyone.


| Model | Resolution | Text-aligned | Checkpoint | Date | Note |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `VGGT-Omega-1B-512` | 512 | No | [Download](https://huggingface.co/facebook/VGGT-Omega/blob/main/vggt_omega_1b_512.pt) | <sub>May 2026</sub> | <sub>Recommended for in-the-wild applications, no benchmarking.</sub> |
| `VGGT-Omega-1B-416-Reproduction` | 416 | No | [Download](https://huggingface.co/facebook/VGGT-Omega/blob/main/vggt_omega_1b_416_reproduce.pt) | <sub>Sep 2026</sub> | <sub>Retrained for benchmarking, as detailed in [reproduction.md](reproduction.md).</sub> |
| `VGGT-Omega-1B-256-Text-Alignment` | 256 | Yes | [Download](https://huggingface.co/facebook/VGGT-Omega/blob/main/vggt_omega_1b_256_text.pt) | <sub>May 2026</sub> | |



## Quick Start

First, clone this repository and install the dependencies:

```bash
git clone git@github.com:facebookresearch/vggt-omega.git
cd vggt-omega
pip install -r requirements.txt
pip install -e .
```


Now, try the model with a few lines of code:

```python
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

checkpoint_path = "path/to/vggt_omega_1b_512.pt"
image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]

model = VGGTOmega().to("cuda").eval()
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

images = load_and_preprocess_images(image_names, image_resolution=512).to("cuda")

with torch.inference_mode():
    predictions = model(images)

extrinsics, intrinsics = encoding_to_camera(
    predictions["pose_enc"],
    predictions["images"].shape[-2:],
)

depth = predictions["depth"]
depth_conf = predictions["depth_conf"]
camera_and_register_tokens = predictions["camera_and_register_tokens"]
camera_tokens = camera_and_register_tokens[:, :, :1]
registers = camera_and_register_tokens[:, :, 1:]
```

For the text-aligned checkpoint, use `VGGTOmega(enable_alignment=True)` with `image_resolution=256` and read `predictions["text_alignment_embedding"]`.


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

## Runtime and GPU Memory

We benchmark the end-to-end peak GPU memory usage of `VGGT-Omega-1B-512` on a
single NVIDIA A100 GPU with 624x416 input images. The measurement covers the full
inference program, from loading the model weights onto the GPU through the
forward pass, so it includes both the memory needed to store the model itself
and the memory used by inference activations and buffers. In other words, a GPU
with at least the listed available memory is able to run the corresponding
number of input frames under this setup.

| **Input Frames** | 1 | 10 | 25 | 50 | 100 | 200 | 300 | 400 | 500 |
|:----------------:|:-:|:--:|:--:|:--:|:---:|:---:|:---:|:---:|:---:|
| **Peak Memory (GB)** | 6.02 | 6.67 | 7.80 | 9.66 | 13.37 | 20.82 | 28.26 | 35.71 | 43.15 |

The benchmark uses [`load_and_preprocess_images`](./vggt_omega/utils/load_fn.py)
with the default `mode="balanced"` and `image_resolution=512`. For these roughly
3:2 landscape images, this produces 624x416 inputs. You can set
`mode="max_size"` to resize the longest side to 512 instead; for the same aspect
ratio, this gives about 512x336 inputs and uses less GPU memory.

## License

See the [LICENSE](./LICENSE) file for details about the license under which
this code is made available.

[^release]: This Release is intended to support the open source research community.

```bibtex
@misc{wang2026vggtomega,
      title={VGGT-$\Omega$},
      author={Jianyuan Wang and Minghao Chen and Shangzhan Zhang and Nikita Karaev and Johannes Schönberger and Patrick Labatut and Piotr Bojanowski and David Novotny and Andrea Vedaldi and Christian Rupprecht},
      year={2026},
      eprint={2605.15195},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.15195},
}
```
