# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from datetime import timedelta

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def get_machine_local_and_dist_rank():
    """Read this process's local and global rank from the torchrun environment."""
    local_rank = os.environ.get("LOCAL_RANK")
    distributed_rank = os.environ.get("RANK")
    assert (
        local_rank is not None and distributed_rank is not None
    ), "LOCAL_RANK and RANK must be set. Launch with torchrun."
    return int(local_rank), int(distributed_rank)


def setup_distributed_backend(backend, timeout_mins):
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_mins))


def unwrap_ddp_if_wrapped(model):
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model
