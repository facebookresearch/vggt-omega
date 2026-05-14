# Inference

This page shows the basic camera, depth, and token inference path.

## Camera and Depth

```python
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import pose_encoding_to_extri_intri

device = "cuda"
image_paths = [
    "examples/znz_20260430_6_crop_top100_00.png",
    "examples/znz_20260430_6_crop_top100_01.png",
    "examples/znz_20260430_6_crop_top100_02.png",
]

model = VGGTOmega().to(device).eval()
model.load_state_dict(
    torch.load("checkpoints/VGGT-Omega-1B-512/model.pt", map_location="cpu")
)

images = load_and_preprocess_images(image_paths, image_resolution=512).to(device)

with torch.inference_mode():
    predictions = model(images)

extrinsics, intrinsics = pose_encoding_to_extri_intri(
    predictions["pose_enc"],
    predictions["images"].shape[-2:],
)

depth = predictions["depth"]
depth_conf = predictions["depth_conf"]
```

`extrinsics` are camera-from-world matrices in OpenCV coordinates. `intrinsics`
assume the principal point is at the image center, matching the released camera
encoding.

## Output Shapes

For input images shaped `[S, 3, H, W]`, the model returns:

| Key | Shape |
| --- | --- |
| `pose_enc` | `[1, S, 9]` |
| `depth` | `[1, S, H, W, 1]` |
| `depth_conf` | `[1, S, H, W]` |
| `camera_and_register_tokens` | `[1, S, 17, 2048]` |
| `images` | `[1, S, 3, H, W]` |

The first token in `camera_and_register_tokens` is the camera token:

```python
camera_tokens = predictions["camera_and_register_tokens"][:, :, :1]
registers = predictions["camera_and_register_tokens"][:, :, 1:]
```

## Text Alignment

The 256-resolution language-aligned checkpoint adds two output keys:

```python
model = VGGTOmega(enable_alignment=True).to(device).eval()
model.load_state_dict(
    torch.load(
        "checkpoints/VGGT-Omega-1B-256-Text-Alignment/model.pt",
        map_location="cpu",
    )
)

images = load_and_preprocess_images(image_paths, image_resolution=256).to(device)

with torch.inference_mode():
    predictions = model(images)

text_embedding = predictions["text_alignment_embedding"]
text_token = predictions["text_alignment_token"]
```

The alignment head reads from the camera/register tokens. It does not require a
language model at inference time.

## Precision

The backbone and aggregator run under autocast with bfloat16 when supported,
otherwise float16. The camera, depth, and alignment heads run in float32.
