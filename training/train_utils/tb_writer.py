# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import atexit
import logging
import uuid
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter

from .distributed import get_machine_local_and_dist_rank


class TensorBoardLogger:
    """Writes scalars from rank 0 only. Calls on other ranks are no-ops."""

    def __init__(self, path: str, filename_suffix: Optional[str] = None, **kwargs: Any) -> None:
        self._writer: Optional[SummaryWriter] = None
        self._path = path
        _, rank = get_machine_local_and_dist_rank()
        if rank == 0:
            logging.info(f"TensorBoard files will be written to {path}")
            self._writer = SummaryWriter(log_dir=path, filename_suffix=filename_suffix or str(uuid.uuid4()), **kwargs)
        atexit.register(self.close)

    @property
    def path(self) -> str:
        return self._path

    def log(self, name: str, data: Any, step: int) -> None:
        if self._writer:
            self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def log_dict(self, payload: Dict[str, Any], step: int) -> None:
        for key, value in payload.items():
            self.log(key, value, step)

    def flush(self) -> None:
        if self._writer:
            self._writer.flush()

    def close(self) -> None:
        if self._writer:
            self._writer.close()
            self._writer = None
