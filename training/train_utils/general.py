# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
from collections import defaultdict
from dataclasses import fields, is_dataclass
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

import numpy as np
import torch


def makedir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def get_resume_checkpoint(checkpoint_save_dir):
    """Return the checkpoint to auto-resume from, or None.

    DDP writes a file, `checkpoint.pt`; FSDP2 writes a DCP directory, `checkpoint/`.
    Checking only for the file makes an FSDP run silently restart from scratch on every
    requeue, since it never recognises the checkpoint it just wrote.
    """
    ckpt_file = os.path.join(checkpoint_save_dir, "checkpoint.pt")
    if os.path.isfile(ckpt_file):
        return ckpt_file
    ckpt_dir = os.path.join(checkpoint_save_dir, "checkpoint")
    if os.path.isdir(ckpt_dir):
        return ckpt_dir
    return None


def get_amp_type(amp_type: Optional[str] = None):
    if amp_type is None:
        return None
    if amp_type not in ("bfloat16", "float16"):
        raise ValueError(f"Invalid amp type: {amp_type}")
    return torch.bfloat16 if amp_type == "bfloat16" else torch.float16


def set_seeds(seed_value, max_epochs, dist_rank):
    """Give every rank a different seed, so ranks do not draw identical data orders."""
    seed_value = (seed_value + dist_rank) * max_epochs
    logging.info(f"Seed for this rank: {seed_value}")
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def check_and_fix_inf_nan(loss_tensor, loss_name, hard_max=None):
    """Replace inf/nan with zero, and optionally clamp to +/- hard_max.

    Used inside the loss to keep a single bad sample from poisoning the whole batch.
    """
    if torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any():
        logging.warning(f"{loss_name} contains inf or nan; setting those entries to 0")
        loss_tensor = torch.where(
            torch.isnan(loss_tensor) | torch.isinf(loss_tensor),
            torch.zeros_like(loss_tensor),
            loss_tensor,
        )

    if hard_max is not None:
        loss_tensor = torch.clamp(loss_tensor, min=-hard_max, max=hard_max)

    return loss_tensor


def human_readable_time(time_seconds):
    minutes, _ = divmod(int(time_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{days:02}d {hours:02}h {minutes:02}m"


@runtime_checkable
class _CopyableData(Protocol):
    def to(self, device: torch.device, *args: Any, **kwargs: Any): ...


def _is_named_tuple(x) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_asdict") and hasattr(x, "_fields")


def copy_data_to_device(data, device: torch.device, *args: Any, **kwargs: Any):
    """Recursively move tensors inside containers, named tuples and dataclasses to a device."""
    if _is_named_tuple(data):
        return type(data)(**copy_data_to_device(data._asdict(), device, *args, **kwargs))
    if isinstance(data, (list, tuple)):
        return type(data)(copy_data_to_device(e, device, *args, **kwargs) for e in data)
    if isinstance(data, defaultdict):
        return type(data)(
            data.default_factory,
            {k: copy_data_to_device(v, device, *args, **kwargs) for k, v in data.items()},
        )
    if isinstance(data, Mapping) and not is_dataclass(data):
        return type(data)({k: copy_data_to_device(v, device, *args, **kwargs) for k, v in data.items()})
    if is_dataclass(data) and not isinstance(data, type):
        moved = type(data)(
            **{
                field.name: copy_data_to_device(getattr(data, field.name), device, *args, **kwargs)
                for field in fields(data)
                if field.init
            }
        )
        for field in fields(data):
            if not field.init:
                setattr(moved, field.name, copy_data_to_device(getattr(data, field.name), device, *args, **kwargs))
        return moved
    if isinstance(data, _CopyableData):
        return data.to(device, *args, **kwargs)
    return data


class AverageMeter:
    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}: {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class DurationMeter:
    def __init__(self, name):
        self.name = name
        self.val = 0

    def update(self, val):
        self.val = val

    def __str__(self):
        return f"{self.name}: {human_readable_time(self.val)}"


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info(" | ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
