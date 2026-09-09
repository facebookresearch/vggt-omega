# Training

Training code for VGGT-&Omega;. This codebase is an agent-assisted cleanup of our original training implementation. We verified parity with the original code over a few training steps, although have not yet validated it with a complete training run. If you encounter any problems, please
open a [GitHub issue](https://github.com/facebookresearch/vggt-omega/issues).

## TODO

- [x] The reannotated UCo3D dataset used to train VGGT-&Omega; is currently being uploaded to [Hugging Face](https://huggingface.co/datasets/facebook/uco3d/tree/main/vggt_omega_anno).
- [x] Add a [supervised geometric filtering pipeline](curation/README.md).
- [x] Add instructions for collecting, converting, and cleaning public datasets.
- [x] Add an agent-assisted visual tool for checking converted and loader-processed data.

## 1. Install

```bash
pip install -r requirements.txt
pip install -e ".[train]"
```

The first command installs the base runtime dependencies for the project. The second installs this
repository in editable mode with the optional `train` dependency group, so local source changes
take effect without reinstalling the package.

The `train` extra adds `hydra-core`, `omegaconf`, `fvcore`, `tensorboard` and `wcmatch`.

## 2. Data Format

The dataloader expects datasets to use a one-directory-per-scene layout:

```
<DATASET_DIR>/<scene>/
  images/              00000.png, 00001.png, ...     names listed in image_names.json
  depths/              00000.exr, 00001.exr, ...     stem matches the image
  image_names.json     list of image file names
  cam_from_worlds.npy  N x 3 x 4, OpenCV, camera-from-world
  intrinsics.npy       N x 3 x 3, OpenCV, in pixels
  ranking.npy          N x M integer neighbor ranking, optional
  <mask_dir>/          00000.png, ..., optional binary masks; 0 marks invalid pixels (see below)
```

The order in `image_names.json` defines the shared frame index for images, depths, poses,
intrinsics, and rankings. Each depth must be a single-channel, floating-point optical-axis
z-depth map aligned pixel-for-pixel with its image; zero marks invalid depth. Metric datasets
should use metres, and depth values and camera translations must always use the same scale.
Cameras are undistorted pinhole cameras with OpenCV axes and `camera-from-world` extrinsics.

Each row of `ranking.npy` starts with its own frame index followed by neighbor indices in ranked
order; `M` may be smaller than `N`. Frame sampling uses this file when `sample_by_index: false`.
If it is missing, sampling falls back to nearby indices in `image_names.json`; with
`sample_by_index: true`, it always samples by index proximity. See
[`unified.py`](data/datasets/unified.py) for details.

Masks are per-dataset and opt-in. Set `mask_path` on a dataset entry to name its mask directory;
depth is zeroed wherever a mask pixel is 0. A mask that is missing or unreadable is warned about
and skipped, and a mask whose resolution differs from the depth map is resized with
nearest-neighbour. Prepared datasets should nevertheless provide one correctly sized mask for every
frame whenever a mask directory is configured.

`sequence_list_file` is an optional allowlist of complete scene-directory names. It selects scenes,
not individual frames.

### Preparing a public dataset

The [dataset preparation guide](dataset_preparation/README.md) explains three agent-facing workflows:
[download and convert](dataset_preparation/download_and_convert.md),
[cleaning](dataset_preparation/clean.md), and
[visual review](dataset_preparation/visualize_and_review.md). The visual-review workflow uses a
[Python template](dataset_preparation/visualize_dataset_template.py) to export real loader samples
as RGB, depth, and GLB assets in a static review site. Start Claude Code or Codex from the repository
root and give it the path to the appropriate prompt together with the required dataset input; the
prompt does not need to be copied into the conversation.

The optional [Supervised geometric filtering](curation/README.md) pipeline reads the same layout.
Users need to provide their own binary labels to train the filtering model.

Then point the config at it:

```yaml
data:
  train:
    dataset:
      dataset_configs:
        uco3d:
          UNIFIED_DIR: /your/path/to/unified/uco3d
```

## 3. Run

The default config is designed for 128 GPUs (16 nodes with 8 GPUs each). To initialize from a
pretrained VGGT-&Omega; checkpoint:

```bash
torchrun \
  --nnodes=16 \
  --nproc_per_node=8 \
  launch.py \
  --config default \
  checkpoint.model_weight_path=/path/to/vggt_omega_checkpoint.pt
```

If you do not use a pretrained VGGT-&Omega; checkpoint, you must instead initialize the image
backbone and aggregator blocks from a DINOv3 checkpoint. This initialization is essential for achieving the expected performance.


```bash
torchrun \
  --nnodes=16 \
  --nproc_per_node=8 \
  launch.py \
  --config default \
  checkpoint.dinov3_weight_path=/path/to/dinov3_vitl16_checkpoint.pth
```

`--config NAME` loads `config/NAME.yaml`; omit the `.yaml` suffix. Additional `key=value`
arguments override config values. Node-rank and rendezvous settings are cluster-specific.

## 4. Distributed Training and Checkpoints

Training is GPU-only. NCCL is the default distributed backend, and the supported strategies are
DDP (`strategy: ddp`) and FSDP2 (`strategy: fsdp`). To enable FSDP2 with its built-in defaults:

```yaml
strategy: fsdp
fsdp_settings: {}
accum_steps: 1
```

If `fsdp_settings.param_dtype` is `bfloat16`
or `float16`, AMP must be enabled and `optim.amp.amp_dtype` must match it. The default FSDP2
settings are defined by [`FSDPSettings`](train_utils/fsdp.py).

If checkpoint.save_dir contains checkpoint.pt (DDP) or checkpoint/ (FSDP2), training resumes automatically. Otherwise, the DINOv3 backbone plus aggregator and/or model are initialized from checkpoint.dinov3_weight_path and checkpoint.model_weight_path respectively; if both are null, training starts from random weights.

DDP checkpoints are single `.pt` files. FSDP2 checkpoints are sharded DCP directories; convert
one to an inference-compatible `.pt` file with:

```bash
python consolidate_checkpoint.py logs/exp/checkpoints/checkpoint checkpoint.pt
```



## 5. Memory

If training runs out of GPU memory, try the following, in order:

1. Enable FSDP. It is disabled by default.
2. Lower **`data.train.per_gpu_batch_scale`** (default: `1.0`, means using 30-40 frames per GPU per iteration). This scales the per-GPU batch size.
3. Reduce the upper bound of **`data.train.common_config.frames_per_sample_range`** (default: `[2, 32]`). Fewer frames per sample reduce activation memory.

Setting `use_checkpoint: true` enables activation checkpointing for the aggregator and patch embedding layers. Keep it enabled unless you have memory to spare.
