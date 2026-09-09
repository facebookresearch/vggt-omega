# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, Dict, List

import torch
import torch.nn as nn


def robust_torch_save(checkpoint: Dict[str, Any], checkpoint_path: str) -> None:
    """Save via a temporary file and an atomic rename, so a preempted save cannot corrupt
    or leave behind the previous checkpoint."""
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, "wb") as f:
        torch.save(checkpoint, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, checkpoint_path)


class DDPCheckpointSaver:
    """Writes one .pt per checkpoint name from rank 0, since DDP ranks hold identical weights."""

    def __init__(self, checkpoint_folder: str, checkpoint_names: List[str], rank: int, epoch: int):
        self.checkpoint_folder = checkpoint_folder
        self.checkpoint_names = checkpoint_names
        self.rank = rank
        self.epoch = epoch

    def save_checkpoint(self, model: nn.Module, **kwargs: Any) -> None:
        if self.rank != 0:
            return
        checkpoint = dict(**kwargs)
        checkpoint["model"] = model.state_dict()
        for name in self.checkpoint_names:
            path = os.path.join(self.checkpoint_folder, f"{name}.pt")
            logging.info(f"Saving checkpoint at epoch {self.epoch} to {path}")
            robust_torch_save(checkpoint, path)


def load_checkpoint(path: str, map_location="cpu") -> Dict[str, Any]:
    logging.info(f"Loading checkpoint from {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def load_model_weights(model: nn.Module, path: str, strict: bool = False) -> None:
    """Initialise weights from a checkpoint without restoring optimizer or epoch state.

    Reads the 'model' entry if present, otherwise treats the file as a bare state dict, which
    is what the released inference checkpoints are.
    """
    checkpoint = load_checkpoint(path)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    result = model.load_state_dict(state_dict, strict=strict)
    missing, unexpected = getattr(result, "missing_keys", []), getattr(result, "unexpected_keys", [])
    logging.info(f"Loaded weights from {path}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if missing:
        logging.info(f"Missing keys (first 20): {missing[:20]}")
    if unexpected:
        logging.info(f"Unexpected keys (first 20): {unexpected[:20]}")


def _initialize_aggregator_blocks_from_dinov3(aggregator: nn.Module) -> None:
    """Initialise both aggregator block streams from the loaded DINOv3 blocks.

    Blocks are aligned from the end. If the aggregator is deeper than the DINOv3 backbone,
    the source blocks are reused cyclically, matching the original Omega training code.
    ``strict=False`` preserves aggregator-only parameters such as Q/K normalisation.
    """
    source_blocks = getattr(aggregator.patch_embed, "blocks", None)
    frame_blocks = getattr(aggregator, "frame_blocks", None)
    inter_frame_blocks = getattr(aggregator, "inter_frame_blocks", None)
    if source_blocks is None or len(source_blocks) == 0:
        raise ValueError("DINOv3 backbone does not expose any blocks for aggregator initialisation")
    if frame_blocks is None or inter_frame_blocks is None:
        raise ValueError("Aggregator must expose frame_blocks and inter_frame_blocks")

    num_source_blocks = len(source_blocks)
    num_aggregator_blocks = min(len(frame_blocks), len(inter_frame_blocks))
    with torch.no_grad():
        for offset in range(1, num_aggregator_blocks + 1):
            source_index = -((offset - 1) % num_source_blocks + 1)
            source_state_dict = source_blocks[source_index].state_dict()
            frame_blocks[-offset].load_state_dict(source_state_dict, strict=False)
            inter_frame_blocks[-offset].load_state_dict(source_state_dict, strict=False)

    logging.info(
        f"Initialised {num_aggregator_blocks} frame and inter-frame aggregator blocks "
        f"from {num_source_blocks} DINOv3 blocks"
    )


def load_dinov3_weights(model: nn.Module, path: str, strict: bool = True) -> None:
    """Load DINOv3 and use its transformer blocks to initialise the aggregator.

    The checkpoint must be an official bare DINOv3 state dict. After loading the image
    backbone, its blocks initialise both aggregator attention streams.
    """
    aggregator = getattr(model, "aggregator", None)
    target = getattr(aggregator, "patch_embed", None)
    if not isinstance(target, nn.Module):
        raise ValueError("DINOv3 initialisation requires model.aggregator.patch_embed to be an nn.Module")

    state_dict = load_checkpoint(path)
    state_dict = {key: value for key, value in state_dict.items() if "local_cls_norm" not in key}

    try:
        result = target.load_state_dict(state_dict, strict=strict)
    except RuntimeError as error:
        raise RuntimeError(f"DINOv3 checkpoint at {path} is incompatible with the configured backbone") from error

    missing = getattr(result, "missing_keys", [])
    unexpected = getattr(result, "unexpected_keys", [])
    logging.info(f"Loaded DINOv3 weights from {path}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if missing:
        logging.info(f"Missing DINOv3 keys (first 20): {missing[:20]}")
    if unexpected:
        logging.info(f"Unexpected DINOv3 keys (first 20): {unexpected[:20]}")

    _initialize_aggregator_blocks_from_dinov3(aggregator)


def restore_optimizer_state(optimizers: List[Any], state: Any) -> None:
    if state is None:
        return
    states = state if isinstance(state, list) else [state]
    if len(states) != len(optimizers):
        raise ValueError(f"Checkpoint holds {len(states)} optimizer states but there are {len(optimizers)}")
    for optimizer, optimizer_state in zip(optimizers, states):
        optimizer.load_state_dict(optimizer_state)


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device, non_blocking=True)
