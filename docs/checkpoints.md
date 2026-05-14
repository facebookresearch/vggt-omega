# Checkpoints

VGGT-Omega releases two 1B checkpoints.

| Model | Constructor | Image Resolution | Outputs |
| --- | --- | --- | --- |
| `VGGT-Omega-1B-512` | `VGGTOmega()` | `512` | camera, depth, camera/register tokens |
| `VGGT-Omega-1B-256-Text-Alignment` | `VGGTOmega(enable_alignment=True)` | `256` | camera, depth, camera/register tokens, text alignment |

Both checkpoint files are raw PyTorch `state_dict` files. They do not include
optimizer state, trainer state, or Qwen/VLM teacher weights.

## Download

The intended release layout is:

```text
checkpoints/
├── VGGT-Omega-1B-512/
│   └── model.pt
└── VGGT-Omega-1B-256-Text-Alignment/
    └── model.pt
```

Download commands will use the final published checkpoint URLs:

```bash
mkdir -p checkpoints/VGGT-Omega-1B-512
wget -O checkpoints/VGGT-Omega-1B-512/model.pt \
  https://huggingface.co/facebook/VGGT-Omega-1B-512/resolve/main/model.pt

mkdir -p checkpoints/VGGT-Omega-1B-256-Text-Alignment
wget -O checkpoints/VGGT-Omega-1B-256-Text-Alignment/model.pt \
  https://huggingface.co/facebook/VGGT-Omega-1B-256-Text-Alignment/resolve/main/model.pt
```

If the hosting location changes before public release, only the URLs above need
to be updated.

## Loading

```python
import torch
from vggt_omega.models import VGGTOmega

model = VGGTOmega().cuda().eval()
state_dict = torch.load("checkpoints/VGGT-Omega-1B-512/model.pt", map_location="cpu")
model.load_state_dict(state_dict)
```

For the language-aligned checkpoint:

```python
model = VGGTOmega(enable_alignment=True).cuda().eval()
state_dict = torch.load(
    "checkpoints/VGGT-Omega-1B-256-Text-Alignment/model.pt",
    map_location="cpu",
)
model.load_state_dict(state_dict)
```

## Differences Between Checkpoints

The two checkpoints share the same camera head, dense depth head, and
aggregator architecture. The 256 checkpoint additionally includes a lightweight
text-alignment head that reads from the camera/register tokens.

The text-alignment release does not instantiate or require the Qwen/VLM teacher
used during training.
