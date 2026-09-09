# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import operator
import random
from dataclasses import dataclass

import numpy as np


STATIC_FRAME_WINDOW_POLICY = "static"
FRAME_COUNT_POWER_LAW_POLICY = "frame_count_power_law"
_DATASET_FRAME_WINDOW_POLICIES = {
    STATIC_FRAME_WINDOW_POLICY,
    FRAME_COUNT_POWER_LAW_POLICY,
}
REVISIT_POLICY_ALLOW = "allow"
REVISIT_POLICY_AVOID_IF_POSSIBLE = "avoid_if_possible"
REVISIT_POLICIES = frozenset({REVISIT_POLICY_ALLOW, REVISIT_POLICY_AVOID_IF_POSSIBLE})


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        ) from error
    if value < minimum:
        raise ValueError(
            f"{name} must be greater than or equal to {minimum}, got {value}"
        )
    return value


def _finite_float(value, name, minimum, maximum=None):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}, got {value}")
    return value


@dataclass(frozen=True)
class FrameCountPowerLawConfig:
    full_window_frame_count: int = 9
    window_scale_exponent: float = 0.5
    min_window_scale: float = 0.5

    def __post_init__(self):
        object.__setattr__(
            self,
            "full_window_frame_count",
            _positive_int(
                self.full_window_frame_count,
                "frame_count_power_law_defaults.full_window_frame_count",
                minimum=2,
            ),
        )
        object.__setattr__(
            self,
            "window_scale_exponent",
            _finite_float(
                self.window_scale_exponent,
                "frame_count_power_law_defaults.window_scale_exponent",
                minimum=0.0,
            ),
        )
        if self.window_scale_exponent == 0:
            raise ValueError(
                "frame_count_power_law_defaults.window_scale_exponent must be greater than 0"
            )
        object.__setattr__(
            self,
            "min_window_scale",
            _finite_float(
                self.min_window_scale,
                "frame_count_power_law_defaults.min_window_scale",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        if self.min_window_scale == 0:
            raise ValueError(
                "frame_count_power_law_defaults.min_window_scale must be greater than 0"
            )

    @classmethod
    def from_config(cls, config):
        if config is None:
            return cls()
        if hasattr(config, "items"):
            values = dict(config.items())
        elif hasattr(config, "__dict__"):
            values = {
                key: getattr(config, key)
                for key in (
                    "full_window_frame_count",
                    "window_scale_exponent",
                    "min_window_scale",
                )
                if hasattr(config, key)
            }
        else:
            raise ValueError(
                "frame_count_power_law_defaults must be a mapping or config object"
            )
        known = {
            "full_window_frame_count",
            "window_scale_exponent",
            "min_window_scale",
        }
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(
                "Unknown frame_count_power_law_defaults keys: " + ", ".join(unknown)
            )
        return cls(**values)


def validate_dataset_frame_window_policy(frame_window_policy):
    if frame_window_policy is None:
        return None
    if (
        not isinstance(frame_window_policy, str)
        or frame_window_policy not in _DATASET_FRAME_WINDOW_POLICIES
    ):
        supported = ", ".join(sorted(_DATASET_FRAME_WINDOW_POLICIES))
        raise ValueError(
            f"Unknown frame_window_policy {frame_window_policy!r}; expected one of: {supported}"
        )
    return frame_window_policy


def resolve_frame_window(
    base_frame_window,
    num_images,
    *,
    frame_window_policy=None,
    low_n_config=None,
):
    """Resolve the candidate range without consuming RNG state."""
    base_frame_window = _positive_int(base_frame_window, "base_frame_window")
    frame_window_policy = validate_dataset_frame_window_policy(frame_window_policy)
    if frame_window_policy != FRAME_COUNT_POWER_LAW_POLICY:
        return base_frame_window

    num_images = _positive_int(num_images, "num_images")
    config = (
        low_n_config
        if isinstance(low_n_config, FrameCountPowerLawConfig)
        else FrameCountPowerLawConfig.from_config(low_n_config)
    )
    transition_ratio = (num_images - 1) / (config.full_window_frame_count - 1)
    scale = max(
        config.min_window_scale,
        min(1.0, transition_ratio**config.window_scale_exponent),
    )
    return max(1, math.floor(base_frame_window * scale + 0.5))


class LazyIndexRanking:
    """Generate index-ranking rows on demand without materializing a matrix."""

    def __init__(self, num_frames, frame_window):
        self._num_frames = _positive_int(num_frames, "num_frames")
        frame_window = _positive_int(frame_window, "frame_window")

        max_neighbors_to_generate = 3 * frame_window + 3
        self._num_neighbors = min(max_neighbors_to_generate, self._num_frames)
        indices = np.arange(2 * max_neighbors_to_generate, dtype=np.int64)
        self._offsets = np.where(
            indices % 2 == 0,
            indices // 2,
            -(indices + 1) // 2,
        )

    @property
    def shape(self):
        return self._num_frames, self._num_neighbors

    def __getitem__(self, row):
        row = operator.index(row)
        if row < 0:
            row += self._num_frames
        if row < 0 or row >= self._num_frames:
            raise IndexError(
                f"ranking row index {row} is out of bounds for {self._num_frames} frames"
            )

        candidates = row + self._offsets
        valid = (candidates >= 0) & (candidates < self._num_frames)
        return candidates[valid][: self._num_neighbors]


def validate_revisit_policy(revisit_policy):
    if not isinstance(revisit_policy, str) or revisit_policy not in REVISIT_POLICIES:
        choices = ", ".join(sorted(REVISIT_POLICIES))
        raise ValueError(
            f"revisit_policy must be one of {{{choices}}}, got {revisit_policy!r}"
        )
    return revisit_policy


def validate_revisit_avoid_prob(revisit_avoid_prob):
    probability = _finite_float(
        revisit_avoid_prob,
        "revisit_avoid_prob",
        minimum=0.0,
        maximum=1.0,
    )
    return probability


def sample_frames_with_ranking(
    initial_id,
    ranking,
    num_samples,
    frame_window,
    jump_prob=-1.0,
    seq_name=None,
    revisit_policy=REVISIT_POLICY_ALLOW,
):
    """Run a ranked moving-anchor walk with optional revisit avoidance."""
    validate_revisit_policy(revisit_policy)
    avoid_revisit = revisit_policy == REVISIT_POLICY_AVOID_IF_POSSIBLE

    if ranking.shape[1] <= 1:
        logging.warning(
            "Ranking has only one neighbor for %s in sequence %s",
            initial_id,
            seq_name,
        )
        return np.full(num_samples, initial_id)

    jump_range = int(1.5 * frame_window)
    sampled_ids = [initial_id]
    visited = {int(initial_id)} if avoid_revisit else None
    current_id = initial_id
    alpha = num_samples / 128.0

    available_neighbors = ranking.shape[1] - 1
    normal_neighbor_count = min(available_neighbors, frame_window)
    normal_weights = [(index + 1) ** alpha for index in range(normal_neighbor_count)]
    if jump_prob > 0:
        jump_neighbor_count = min(available_neighbors, jump_range)
        jump_weights = [(index + 1) ** alpha for index in range(jump_neighbor_count)]
    else:
        jump_weights = None

    for _ in range(num_samples - 1):
        if jump_prob > 0 and random.random() < jump_prob:
            neighbors = ranking[current_id][1 : jump_range + 1]
            weights = jump_weights
        else:
            neighbors = ranking[current_id][1 : frame_window + 1]
            weights = normal_weights

        if avoid_revisit:
            fresh_indices = [
                index
                for index, neighbor in enumerate(neighbors)
                if int(neighbor) not in visited
            ]
            if fresh_indices:
                neighbors = [neighbors[index] for index in fresh_indices]
                weights = [weights[index] for index in fresh_indices]

        next_id = random.choices(neighbors, weights=weights, k=1)[0]
        sampled_ids.append(next_id)
        if avoid_revisit:
            visited.add(int(next_id))
        current_id = next_id

    return np.asarray(sampled_ids)
