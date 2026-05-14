# Camera and Register Tokens

VGGT-Omega appends one camera token and sixteen register tokens to each frame.
The paper also refers to the registers as scene tokens because they aggregate
scene-level information across the input views.

At inference time, `VGGTOmega.forward()` returns:

```python
tokens = predictions["camera_and_register_tokens"]
```

with shape:

```text
[B, S, 17, 2048]
```

The token layout is:

```python
camera_tokens = tokens[:, :, :1]
registers = tokens[:, :, 1:]
```

## Why Return These Tokens

The camera token is used by the camera head to predict pose and field of view.
The registers are used by the model to collect information across frames. The
language-alignment head also reads from the camera/register tokens rather than
from dense image patch tokens.

Returning these tokens makes the release useful beyond camera and depth
prediction. They can be used as compact sequence-level or frame-level features
for downstream spatial understanding tasks.

## Register Attention

VGGT-Omega replaces a subset of full inter-frame attention layers with register
attention. In those layers, inter-frame communication is restricted to the
registers. The updated registers then interact with image tokens in subsequent
frame attention layers.

This design keeps the model simple while encouraging the registers to carry
global scene information.

## Language Alignment Checkpoint

The `VGGT-Omega-1B-256-Text-Alignment` checkpoint adds a small alignment head.
It introduces a learnable language token, lets it attend to the camera/register
tokens, and projects the result into an embedding:

```python
text_embedding = predictions["text_alignment_embedding"]
text_token = predictions["text_alignment_token"]
```

The released inference code does not include the Qwen/VLM teacher used during
training.
