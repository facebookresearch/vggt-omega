# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC licence,
# found in the LICENSE_CELL_DINO_CODE file in the root directory of this source tree.

from typing import Any

from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
from torch import nn

import dinov2.distributed as dist


class PeriodicCheckpointerWithCleanup(PeriodicCheckpointer):
    @property
    def does_write(self) -> bool:
        """See https://github.com/facebookresearch/fvcore/blob/main/fvcore/common/checkpoint.py#L114"""
        return self.checkpointer.save_dir and self.checkpointer.save_to_disk

    def save_best(self, **kwargs: Any) -> None:
        """Same argument as `Checkpointer.save`, to save a model named like `model_best.pth`"""
        self.checkpointer.save(f"{self.file_prefix}_best", **kwargs)

    def has_checkpoint(self) -> bool:
        return self.checkpointer.has_checkpoint()

    def get_checkpoint_file(self) -> str:  # returns "" if the file does not exist
        return self.checkpointer.get_checkpoint_file()

    def load(self, path: str, checkpointables=None) -> dict[str, Any]:
        return self.checkpointer.load(path=path, checkpointables=checkpointables)

    def step(self, iteration: int, **kwargs: Any) -> None:
        if not self.does_write:  # step also removes files, so should be deactivated when object does not write
            return
        super().step(iteration=iteration, **kwargs)


def resume_or_load(checkpointer: Checkpointer, path: str, *, resume: bool = True) -> dict[str, Any]:
    """
    If `resume` is True, this method attempts to resume from the last
    checkpoint, if exists. Otherwise, load checkpoint from the given path.
    Similar to Checkpointer.resume_or_load in fvcore
    https://github.com/facebookresearch/fvcore/blob/main/fvcore/common/checkpoint.py#L208
    but always reload checkpointables, in case we want to resume the training in a new job.
    """
    if resume and checkpointer.has_checkpoint():
        path = checkpointer.get_checkpoint_file()
    return checkpointer.load(path)


def build_periodic_checkpointer(
    model: nn.Module,
    save_dir="",
    *,
    period: int,
    max_iter=None,
    max_to_keep=None,
    **checkpointables: Any,
) -> PeriodicCheckpointerWithCleanup:
    """Util to build a `PeriodicCheckpointerWithCleanup`."""
    checkpointer = Checkpointer(model, save_dir, **checkpointables, save_to_disk=dist.is_main_process())
    return PeriodicCheckpointerWithCleanup(checkpointer, period, max_iter=max_iter, max_to_keep=max_to_keep)
