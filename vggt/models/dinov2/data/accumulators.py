# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List, Optional, Any

import torch
from torch import Tensor
from torch.nn import functional as F

import torch.distributed as dist
from dinov2.distributed import get_global_size


def _simple_gather_all_tensors(result: torch.Tensor, group: Any, world_size: int) -> List[torch.Tensor]:
    gathered_result = [torch.zeros_like(result) for _ in range(world_size)]
    dist.all_gather(gathered_result, result, group)
    return gathered_result


def gather_all_tensors(result: torch.Tensor, group: Optional[Any] = None) -> List[torch.Tensor]:
    """
    Copied from https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/utilities/distributed.py
    Gather all tensors from several ddp processes onto a list that is broadcasted to all processes.

    Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
    tensors are padded, gathered and then trimmed to secure equal workload for all processes.

    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)

    Return:
        list with size equal to the process group where element i corresponds to result tensor from process i
    """
    # convert tensors to contiguous format
    result = result.contiguous()

    world_size = get_global_size()
    dist.barrier(group=group)

    # if the tensor is scalar, things are easy
    if result.ndim == 0:
        return _simple_gather_all_tensors(result, group, world_size)

    # 1. Gather sizes of all tensors
    local_size = torch.tensor(result.shape, device=result.device)
    local_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(local_sizes, local_size, group=group)
    max_size = torch.stack(local_sizes).max(dim=0).values
    all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)

    # 2. If shapes are all the same, then do a simple gather:
    if all_sizes_equal:
        return _simple_gather_all_tensors(result, group, world_size)

    # 3. If not, we need to pad each local tensor to maximum size, gather and then truncate
    pad_dims = []
    pad_by = (max_size - local_size).detach().cpu()
    for val in reversed(pad_by):
        pad_dims.append(0)
        pad_dims.append(val.item())
    result_padded = F.pad(result, pad_dims)
    gathered_result = [torch.zeros_like(result_padded) for _ in range(world_size)]
    dist.all_gather(gathered_result, result_padded, group)
    for idx, item_size in enumerate(local_sizes):
        slice_param = [slice(dim_size) for dim_size in item_size]
        gathered_result[idx] = gathered_result[idx][slice_param]
    return gathered_result


def _cat_and_gather_tensor_list(tensor_list: List[Tensor]) -> Tensor:
    local_cat = torch.cat(tensor_list)
    return torch.cat(gather_all_tensors(local_cat))


class Accumulator:
    def __init__(self) -> None:
        pass

    def update(self, preds: Tensor, target: Tensor, index: Tensor) -> None:
        raise NotImplementedError

    def accumulate(self) -> Optional[Dict[str, Tensor]]:
        raise NotImplementedError


class NoOpAccumulator(Accumulator):
    def __init__(self) -> None:
        pass

    def update(self, preds: Tensor, target: Tensor, index: Tensor) -> None:
        pass

    def accumulate(self) -> None:
        return None


class ResultsAccumulator(Accumulator):
    """
    Accumulate predictions and targets across processes
    """

    def __init__(self) -> None:
        self._local_values: Dict[str, List[Tensor]] = defaultdict(list)
        self._gathered_values: Dict[str, Tensor] = {}
        self._gathered = False

    def update(self, preds: Tensor, target: Tensor, index: Tensor) -> None:
        assert len(preds) == len(target) == len(index)
        assert not self._gathered, "Tensors have already been gathered in this helper"
        self._local_values["preds"].append(preds)
        self._local_values["target"].append(target)
        self._local_values["index"].append(index)
        self._gathered = False

    def _gather_tensors(self):
        for k, tensor_list in self._local_values.items():
            self._gathered_values[k] = _cat_and_gather_tensor_list(tensor_list)
        self._gathered = True

    def accumulate(self) -> Dict[str, Tensor]:
        if not self._gathered:
            self._gather_tensors()
        preds, target, index = [self._gathered_values[k] for k in ["preds", "target", "index"]]
        assert len(preds) == len(target) == len(index) and index.min() == 0
        preds_ordered = torch.zeros((index.max() + 1, *preds.shape[1:]), dtype=preds.dtype, device=preds.device)
        preds_ordered[index] = preds
        target_ordered = torch.zeros((index.max() + 1, *target.shape[1:]), dtype=target.dtype, device=target.device)
        target_ordered[index] = target
        return {"preds": preds_ordered, "target": target_ordered}
