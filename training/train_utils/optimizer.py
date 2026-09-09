# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple, Union

import hydra
import torch
import torch.nn as nn
from fvcore.common.param_scheduler import ConstantParamScheduler
from torch import Tensor
from wcmatch import fnmatch

# Same flags the config patterns were written against: case sensitive, '*' matches dots,
# extended globs, and "a|b" splits into two patterns.
GLOB_FLAGS = fnmatch.CASE | fnmatch.DOTMATCH | fnmatch.EXTMATCH | fnmatch.SPLIT

NORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


class OptimizerWrapper:
    """A torch optimizer plus one scheduler per (param group, option) pair.

    Schedulers are driven by `where`, the fraction of training completed, so they are
    resolution-independent: the same config works for any total step count.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, schedulers=None) -> None:
        self.optimizer = optimizer
        self.schedulers = schedulers or [{} for _ in optimizer.param_groups]
        self._validate_optimizer_schedulers()
        self.step_schedulers(0.0)

    def _validate_optimizer_schedulers(self):
        if len(self.schedulers) != len(self.optimizer.param_groups):
            raise ValueError("Expected one scheduler mapping per optimizer parameter group")
        for sched_map in self.schedulers:
            for option in sched_map:
                if option not in self.optimizer.defaults:
                    raise ValueError(
                        f"Optimizer option '{option}' is not one of {list(self.optimizer.defaults)}"
                    )

    def step_schedulers(self, where: float) -> None:
        # fvcore schedulers reject where == 1.0, so the caller must skip the final update.
        for i, param_group in enumerate(self.optimizer.param_groups):
            for option, scheduler in self.schedulers[i].items():
                param_group[option] = scheduler(where)


def get_full_parameter_name(module_name: str, param_name: str) -> str:
    return param_name if module_name == "" else f"{module_name}.{param_name}"


def unix_param_pattern_to_parameter_names(patterns: Optional[List[str]], parameter_names: Set[str]) -> Set[str]:
    if not patterns:
        return set()
    matched = []
    for pattern in patterns:
        hits = set(fnmatch.filter(parameter_names, pattern, flags=GLOB_FLAGS))
        if not hits:
            raise ValueError(f"Parameter pattern '{pattern}' matched nothing")
        logging.info(f"Pattern '{pattern}' matched {len(hits)} parameters")
        matched.append(hits)
    return set.union(*matched)


def _pattern_to_parameter_names(cfg: Dict, parameter_names: Set[str]):
    if "module_cls_names" in cfg:
        raise ValueError("Optimizer scheduler module_cls_names is not supported; use param_names")
    if "param_names" not in cfg:
        return None
    return unix_param_pattern_to_parameter_names(cfg["param_names"], parameter_names)


def set_default_parameters(scheduler_cfgs: List[Dict], all_parameter_names: Set[str]) -> None:
    """Give the one cfg without explicit patterns everything the others did not claim."""
    claimed = [cfg["parameter_names"] for cfg in scheduler_cfgs if cfg["parameter_names"]]
    default_params = all_parameter_names if not claimed else all_parameter_names - set.union(*claimed)

    defaults = 0
    for cfg in scheduler_cfgs:
        if cfg["parameter_names"] is None:
            cfg["parameter_names"] = default_params
            defaults += 1
    if defaults > 1:
        raise ValueError("At most one scheduler per option may omit param_names")
    if defaults == 0:
        scheduler_cfgs.append({"parameter_names": default_params})


class FilterBiasAndBN:
    """Sets weight decay to zero for biases, norm parameters and every other 1D parameter.

    Splits the weight_decay configs so bias/norm parameters get a constant 0.0 schedule and
    everything else keeps the original schedule.
    """

    def __init__(self, default_weight_decay: float = 0.0):
        self.default_weight_decay = default_weight_decay

    def __call__(self, scheduler_cfgs: List[List[Dict]], model: nn.Module) -> List[List[Dict]]:
        no_decay = self._no_decay_parameter_names(model)
        logging.info(f"FilterBiasAndBN: {len(no_decay)} parameters excluded from weight decay")

        wd_idx = next(
            (i for i, cfgs in enumerate(scheduler_cfgs) if cfgs and cfgs[0].get("option") == "weight_decay"),
            None,
        )

        if wd_idx is None:
            all_params = {n for n, p in model.named_parameters() if p.requires_grad}
            scheduler_cfgs.append(
                [
                    {
                        "option": "weight_decay",
                        "parameter_names": no_decay,
                        "scheduler": ConstantParamScheduler(value=0.0),
                    },
                    {
                        "option": "weight_decay",
                        "parameter_names": all_params - no_decay,
                        "scheduler": ConstantParamScheduler(value=float(self.default_weight_decay)),
                    },
                ]
            )
            return scheduler_cfgs

        split = []
        for cfg in scheduler_cfgs[wd_idx]:
            params = cfg.get("parameter_names") or set()
            decayed = params - no_decay
            undecayed = params & no_decay
            if decayed:
                split.append({**cfg, "parameter_names": decayed})
            if undecayed:
                split.append(
                    {
                        **cfg,
                        "option": "weight_decay",
                        "parameter_names": undecayed,
                        "scheduler": ConstantParamScheduler(value=0.0),
                    }
                )
        scheduler_cfgs[wd_idx] = split
        return scheduler_cfgs

    @staticmethod
    def _no_decay_parameter_names(model: nn.Module) -> Set[str]:
        no_decay: Set[str] = set()
        for module_name, module in model.named_modules():
            if isinstance(module, NORM_TYPES):
                for pname, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        no_decay.add(get_full_parameter_name(module_name, pname))
        for name, param in model.named_parameters():
            if param.requires_grad and (name.endswith("bias") or param.ndim == 1):
                no_decay.add(name)
        return no_decay


def validate_param_group_params(param_groups: List[Dict], model: nn.Module) -> None:
    for group in param_groups:
        if len(group["params"]) != len(set(group["params"])):
            raise ValueError("A parameter group contains duplicates")

    grouped = [set(group["params"]) for group in param_groups]
    for a, b in itertools.combinations(grouped, 2):
        if not a.isdisjoint(b):
            raise ValueError("Parameter groups overlap")

    model_params = {p for _, p in model.named_parameters() if p.requires_grad}
    covered = set.union(*grouped) if grouped else set()
    if covered != model_params:
        raise ValueError(f"Parameter groups cover {len(covered)} of {len(model_params)} model parameters")


def name_constraints_to_parameters(constraints: List[Set[str]], named_parameters: Dict[str, Tensor]) -> List[Tensor]:
    matching = set.intersection(*constraints)
    return [v for k, v in named_parameters.items() if k in matching]


def map_scheduler_cfgs_to_param_groups(
    all_scheduler_cfgs: Iterable[List[Dict]], named_parameters: Dict[str, Tensor]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, List[Tensor]]]]:
    """One param group per non-empty combination of one cfg from each option."""
    schedulers: List[Dict[str, Any]] = []
    param_groups: List[Dict[str, List[Tensor]]] = []

    for cfgs in itertools.product(*all_scheduler_cfgs):
        matching = name_constraints_to_parameters([cfg["parameter_names"] for cfg in cfgs], named_parameters)
        if not matching:
            continue
        schedulers.append({cfg["option"]: cfg["scheduler"] for cfg in cfgs if "option" in cfg})
        param_groups.append({"params": matching})

    return schedulers, param_groups


def construct_optimizer(
    model: nn.Module,
    optimizer_conf: Any,
    options_conf: Union[Mapping[str, List], None] = None,
    filter_bias_and_bn: bool = False,
) -> OptimizerWrapper:
    named_parameters = {name: param for name, param in model.named_parameters() if param.requires_grad}
    all_parameter_names = set(named_parameters)

    if not options_conf and not filter_bias_and_bn:
        return OptimizerWrapper(hydra.utils.instantiate(optimizer_conf, named_parameters.values()))

    all_scheduler_cfgs: List[List[Dict]] = []
    for option, cfg_list in hydra.utils.instantiate(options_conf or {}).items():
        cfgs = [dict(cfg) for cfg in cfg_list]
        for cfg in cfgs:
            cfg["option"] = option
            cfg["parameter_names"] = _pattern_to_parameter_names(cfg, all_parameter_names)
        set_default_parameters(cfgs, all_parameter_names)
        all_scheduler_cfgs.append(cfgs)

    if filter_bias_and_bn:
        default_wd = optimizer_conf.get("weight_decay", 0.0)
        all_scheduler_cfgs = FilterBiasAndBN(default_weight_decay=default_wd)(
            scheduler_cfgs=all_scheduler_cfgs,
            model=model,
        )

    schedulers, param_groups = map_scheduler_cfgs_to_param_groups(all_scheduler_cfgs, named_parameters)
    validate_param_group_params(param_groups, model)
    logging.info(f"Constructed {len(param_groups)} optimizer parameter groups")

    return OptimizerWrapper(hydra.utils.instantiate(optimizer_conf, param_groups), schedulers)


def construct_optimizers(model: nn.Module, optim_conf) -> List[OptimizerWrapper]:
    optimizer = construct_optimizer(
        model,
        optim_conf.optimizer,
        getattr(optim_conf, "options", None),
        filter_bias_and_bn=getattr(optim_conf, "filter_bias_and_bn", False),
    )
    return [optimizer]
