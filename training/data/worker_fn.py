# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from functools import partial

import numpy as np
import torch


def default_worker_init_fn(worker_id, epoch, seed=0):
    distributed_rank = int(os.environ["RANK"])
    worker_seed = (
        seed + epoch * 100_003 + distributed_rank * 10_007 + worker_id * 1_009
    ) % (2**32)
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_worker_init_fn(seed, epoch, worker_init_fn=None):
    if worker_init_fn is not None:
        return worker_init_fn
    return partial(default_worker_init_fn, epoch=epoch, seed=seed)
