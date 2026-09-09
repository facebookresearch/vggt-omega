# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Mapping, Sequence

import torch


def is_sequence_of_primitives(data):
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and isinstance(data[0], (str, int, float, bool))
    )


def get_chunk_from_data(data, chunk_id, num_chunks):
    """Recursively split every tensor in `data` into num_chunks and return one chunk."""
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        if len(data) % num_chunks != 0:
            raise ValueError(f"Cannot split length {len(data)} into {num_chunks} equal chunks")
        size = len(data) // num_chunks
        return data[size * chunk_id : size * (chunk_id + 1)]
    if isinstance(data, Mapping):
        return {key: get_chunk_from_data(value, chunk_id, num_chunks) for key, value in data.items()}
    # A str is a Sequence of str, so it has to be handled before the Sequence branch.
    if isinstance(data, str):
        return data
    if isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    return data


def chunk_batch_for_accum_steps(batch, accum_steps: int):
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]
