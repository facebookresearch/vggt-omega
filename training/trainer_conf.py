# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import torch


@dataclass
class TrainerOptimAMPConf:
    enabled: bool = False
    amp_dtype: str = "bfloat16"


@dataclass
class TrainerOptimConf:
    optimizer: torch.optim.Optimizer = None
    frozen_module_names: Optional[List[str]] = None
    frozen_param_names: Optional[List[str]] = None
    options: Optional[Dict[str, Any]] = None
    amp: Optional[Dict[str, Any]] = None
    gradient_clip: Any = None
    filter_bias_and_bn: bool = False

    def __post_init__(self):
        if not isinstance(self.amp, TrainerOptimAMPConf):
            if self.amp is None:
                self.amp = {}
            assert isinstance(self.amp, Mapping)
            self.amp = TrainerOptimAMPConf(**self.amp)


@dataclass
class TrainerDistributedConf:
    backend: Optional[str] = None
    find_unused_parameters: bool = False
    timeout_mins: int = 30
    gradient_as_bucket_view: bool = True
    bucket_cap_mb: int = 25
    broadcast_buffers: bool = True


@dataclass
class TrainerCudaConf:
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    allow_tf32: bool = False


@dataclass
class TrainerCheckpointConf:
    save_dir: str
    save_freq: int
    # Path to a DINOv3 backbone checkpoint used to initialise
    # model.aggregator.patch_embed. Ignored when resuming from save_dir.
    dinov3_weight_path: Optional[str] = None
    dinov3_weight_strict: bool = True
    # Path to a checkpoint whose weights initialise the model, without restoring optimizer
    # or epoch state. Ignored when resuming from save_dir.
    model_weight_path: Optional[str] = None
    model_weight_strict: bool = False


@dataclass
class TrainerLoggingConf:
    log_dir: str
    log_freq: int  # in iterations
    tensorboard_writer: Any
    log_level_primary: str = "INFO"
    log_level_secondary: str = "WARNING"
    scalar_keys_to_log: Optional[Dict[str, Any]] = None
    all_ranks: bool = False
