# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import wraps
from typing import List

import torch.nn as nn
from wcmatch import fnmatch

GLOB_FLAGS = fnmatch.CASE | fnmatch.DOTMATCH | fnmatch.EXTMATCH | fnmatch.SPLIT


def freeze_modules(model: nn.Module, name_patterns: List[str]) -> nn.Module:
    """Freeze modules whose name matches any pattern: no gradients, and held in eval mode.

    Eval mode is enforced by wrapping .train() so a later model.train() cannot undo it.
    """
    matched = []
    for name, module in model.named_modules():
        if not any(fnmatch.fnmatch(name, pattern, flags=GLOB_FLAGS) for pattern in name_patterns):
            continue
        matched.append(name)
        _force_eval(module)
        for param in module.parameters():
            param.requires_grad = False

    _check_patterns_used(matched, name_patterns, "module")
    logging.info(f"Froze {len(matched)} modules: {matched}")
    return model


def freeze_parameters(model: nn.Module, name_patterns: List[str]) -> nn.Module:
    """Freeze parameters whose name matches any pattern. Does not change training mode."""
    matched = []
    for name, param in model.named_parameters():
        if any(fnmatch.fnmatch(name, pattern, flags=GLOB_FLAGS) for pattern in name_patterns):
            matched.append(name)
            param.requires_grad = False

    _check_patterns_used(matched, name_patterns, "parameter")
    logging.info(f"Froze {len(matched)} parameters")
    return model


def _force_eval(module: nn.Module) -> None:
    module.eval()
    original_train = module.train

    @wraps(original_train)
    def locked_train(mode: bool = True):
        return original_train(False)

    module.train = locked_train


def _check_patterns_used(matched_names: List[str], name_patterns: List[str], kind: str) -> None:
    unused = [
        pattern
        for pattern in name_patterns
        if not any(fnmatch.fnmatch(name, pattern, flags=GLOB_FLAGS) for name in matched_names)
    ]
    if unused:
        raise ValueError(f"These frozen {kind} patterns matched nothing: {unused}")
