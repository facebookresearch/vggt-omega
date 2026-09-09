# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch.nn as nn


class GradientClipper:
    """Clips gradients per group of modules.

    Each config entry is {module_name, max_norm, norm_type}, where module_name is a string or
    list of strings matched as substrings against parameter names. Works for DDP and FSDP2:
    clip_grad_norm_ handles the sharded DTensor parameters FSDP2 produces.

    setup_clipping raises if any trainable parameter is left uncovered, so adding a module to
    the model without adding it to a config is an error rather than a silently unclipped group.
    """

    def __init__(self, configs):
        self.configs = []
        self.params_to_clip_by_config: Optional[List] = None

        for config in configs:
            module_names = config["module_name"]
            if isinstance(module_names, str):
                module_names = [module_names]
            self.configs.append(
                {
                    "module_names": module_names,
                    "max_norm": float(config["max_norm"]) if config["max_norm"] is not None else None,
                    "norm_type": config.get("norm_type", 2),
                }
            )

    def setup_clipping(self, model: nn.Module) -> None:
        params_to_clip_by_config = []
        all_clipped = set()

        for config in self.configs:
            group = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if not any(module_name in name for module_name in config["module_names"]):
                    continue
                if param in all_clipped:
                    raise ValueError(f"Parameter '{name}' is matched by more than one clipping config")
                group.append(param)
                all_clipped.add(param)
            params_to_clip_by_config.append((config, group))

        uncovered = [n for n, p in model.named_parameters() if p.requires_grad and p not in all_clipped]
        if uncovered:
            raise ValueError(f"{len(uncovered)} trainable parameters match no clipping config: {uncovered[:10]}")

        self.params_to_clip_by_config = params_to_clip_by_config

    def __call__(self) -> Dict[str, float]:
        if self.params_to_clip_by_config is None:
            raise RuntimeError("Call setup_clipping() before using GradientClipper")

        grad_norms = {}
        for config, params in self.params_to_clip_by_config:
            if not params or config["max_norm"] is None:
                continue
            grad_norm = nn.utils.clip_grad_norm_(
                params, max_norm=config["max_norm"], norm_type=config["norm_type"]
            )
            grad_norms[",".join(config["module_names"])] = grad_norm.item()
        return grad_norms
