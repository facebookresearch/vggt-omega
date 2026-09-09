# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""FSDP2 (fully_shard) support. Requires torch >= 2.6."""

import fnmatch
import inspect
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from .checkpoint import robust_torch_save

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass
class FSDPSettings:
    shard_strategy: str = "FULL_SHARD"
    param_dtype: Optional[str] = "bfloat16"
    reduce_dtype: Optional[str] = "float32"
    buffer_dtype: Optional[str] = "float32"
    cast_root_forward_inputs: Optional[bool] = True
    # Classes to shard individually.
    module_cls_names: Optional[List[str]] = field(default_factory=lambda: ["torch.nn.Linear"])
    # Module names matched with fnmatch and sharded under an fp32 policy before the main pass.
    fp32_module_name_patterns: Optional[List[str]] = field(default_factory=lambda: ["*_head*"])


def to_fsdp_settings(settings: Union[FSDPSettings, Mapping, None]) -> Optional[FSDPSettings]:
    if settings is None or isinstance(settings, FSDPSettings):
        return settings
    return FSDPSettings(**dict(settings))


def _dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if name is None:
        return None
    if name not in DTYPES:
        raise ValueError(f"Unknown dtype '{name}', expected one of {list(DTYPES)}")
    return DTYPES[name]


def _mixed_precision_policy(**kwargs) -> MixedPrecisionPolicy:
    # MixedPrecisionPolicy's signature has changed between torch releases; drop what it
    # does not accept rather than failing on a newer or older torch.
    accepted = set(inspect.signature(MixedPrecisionPolicy).parameters)
    return MixedPrecisionPolicy(**{k: v for k, v in kwargs.items() if k in accepted and v is not None})


def _supports_fully_shard(module: nn.Module) -> bool:
    return type(module).forward is not nn.Module.forward


def _resolve_classes(class_names: List[str]) -> tuple:
    import importlib

    classes = []
    for dotted in class_names:
        module_path, _, cls_name = dotted.rpartition(".")
        if not module_path:
            raise ValueError(f"module_cls_names needs a fully qualified path, got '{dotted}'")
        classes.append(getattr(importlib.import_module(module_path), cls_name))
    return tuple(classes)


def apply_fsdp2_strategy(model: nn.Module, settings: Union[FSDPSettings, Mapping, None]) -> nn.Module:
    settings = to_fsdp_settings(settings)
    if settings is None:
        raise ValueError("fsdp_settings is required when strategy is 'fsdp'")
    if settings.shard_strategy not in {"FULL_SHARD", "SHARD_GRAD_OP"}:
        raise ValueError(
            f"Unknown shard strategy '{settings.shard_strategy}', "
            "expected 'FULL_SHARD' or 'SHARD_GRAD_OP'"
        )

    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    reshard_after_forward = settings.shard_strategy != "SHARD_GRAD_OP"

    policy = _mixed_precision_policy(
        param_dtype=_dtype(settings.param_dtype),
        reduce_dtype=_dtype(settings.reduce_dtype),
        buffer_dtype=_dtype(settings.buffer_dtype),
        cast_forward_inputs=settings.cast_root_forward_inputs,
    )
    fp32_policy = _mixed_precision_policy(
        param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32
    )

    # named_modules() yields parents before children; fully_shard must see children first,
    # so every pass below walks the reversed list.
    all_modules = [(name, module) for name, module in model.named_modules() if name]

    already_wrapped = set()
    fp32_patterns = settings.fp32_module_name_patterns or []
    for name, module in reversed(all_modules):
        if not any(fnmatch.fnmatch(name, pattern) for pattern in fp32_patterns):
            continue
        if not _supports_fully_shard(module):
            continue
        logging.info(f"FSDP2: fully_shard (fp32) {name}")
        fully_shard(module, mesh=mesh, mp_policy=fp32_policy, reshard_after_forward=reshard_after_forward)
        already_wrapped.add(name)

    wrap_classes = _resolve_classes(settings.module_cls_names or [])
    if wrap_classes:
        for name, module in reversed(all_modules):
            if name in already_wrapped or not isinstance(module, wrap_classes):
                continue
            if not _supports_fully_shard(module):
                continue
            fully_shard(module, mesh=mesh, mp_policy=policy, reshard_after_forward=reshard_after_forward)
            already_wrapped.add(name)
        logging.info(f"FSDP2: sharded {len(already_wrapped)} submodules")

    fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=reshard_after_forward)
    return model


def create_grad_scaler(settings: Union[FSDPSettings, Mapping, None], enabled: bool = False):
    if not enabled:
        return torch.amp.GradScaler("cuda", enabled=False)
    settings = to_fsdp_settings(settings)
    if settings is not None and "bfloat16" in (settings.param_dtype, settings.reduce_dtype):
        logging.info("FSDP2: bfloat16 in use, GradScaler disabled")
        return torch.amp.GradScaler("cuda", enabled=False)
    return torch.amp.GradScaler("cuda", enabled=True)


def _sanitize_optimizer_param_groups(optimizers) -> None:
    """OmegaConf containers in param_groups (betas is the usual one) make DCP raise
    'Unexpected value type'. Convert them to plain Python first."""
    try:
        from omegaconf import DictConfig, ListConfig
    except ImportError:
        return
    for optimizer in optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]:
        for group in optimizer.param_groups:
            for key, value in group.items():
                if isinstance(value, ListConfig):
                    group[key] = list(value)
                elif isinstance(value, DictConfig):
                    group[key] = dict(value)


_SD_OPTIONS = StateDictOptions(full_state_dict=False, cpu_offload=True)


def _state_dict_plan(model: nn.Module, optimizers) -> Dict[str, Any]:
    if optimizers:
        _sanitize_optimizer_param_groups(optimizers)
        model_sd, optim_sd = get_state_dict(model, optimizers, options=_SD_OPTIONS)
        return {"model": model_sd, "optimizer": optim_sd}
    model_sd, _ = get_state_dict(model, optimizers=[], options=_SD_OPTIONS)
    return {"model": model_sd}


class ShardedCheckpointSaver:
    """Writes each checkpoint as a DCP directory plus a rank-0 metadata.pt.

    Use consolidate_checkpoint.py to turn one of these directories into a single .pt that
    the inference code can load.
    """

    def __init__(self, checkpoint_folder: str, checkpoint_names: List[str], rank: int):
        self.checkpoint_folder = checkpoint_folder
        self.checkpoint_names = checkpoint_names
        self.rank = rank

    def save_checkpoint(self, model: nn.Module, optimizers=None, **kwargs: Any) -> None:
        state_dict = _state_dict_plan(model, optimizers)
        for name in self.checkpoint_names:
            path = os.path.join(self.checkpoint_folder, name)
            os.makedirs(path, exist_ok=True)
            dcp.save(state_dict=state_dict, checkpoint_id=path)
            if self.rank == 0:
                robust_torch_save(kwargs, os.path.join(path, "metadata.pt"))
            logging.info(f"FSDP2: saved sharded checkpoint to {path}")


class ShardedCheckpointLoader:
    """Loads a DCP directory. Must run after the model is sharded and the optimizer built."""

    def load_checkpoint(self, model: nn.Module, optimizers=None, checkpoint_path: str = "") -> Dict[str, Any]:
        if not checkpoint_path or not os.path.isdir(checkpoint_path):
            logging.warning(f"FSDP2: no sharded checkpoint at {checkpoint_path}")
            return {}

        metadata = {}
        meta_path = os.path.join(checkpoint_path, "metadata.pt")
        if os.path.isfile(meta_path):
            metadata = torch.load(meta_path, map_location="cpu", weights_only=False)

        state_dict = _state_dict_plan(model, optimizers)
        dcp.load(state_dict=state_dict, checkpoint_id=checkpoint_path)
        set_state_dict(
            model,
            optimizers if optimizers else [],
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict.get("optimizer"),
            options=_SD_OPTIONS,
        )
        logging.info(f"FSDP2: loaded sharded checkpoint from {checkpoint_path}")
        return metadata
