# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import logging
import random
from typing import Callable, Optional

import numpy as np
from hydra.utils import instantiate
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from .sampler_configs import FRAME_COUNT_SAMPLING_WEIGHTS, FRAME_COUNT_TO_BATCH_SIZE
from .worker_fn import get_worker_init_fn


DEFAULT_DISCRETE_ASPECT_RATIOS = (
    5.0 / 9.0,
    10.0 / 16.0,
    2.0 / 3.0,
    3.0 / 4.0,
    4.0 / 5.0,
    1.0,
    5.0 / 4.0,
    4.0 / 3.0,
    3.0 / 2.0,
    16.0 / 10.0,
    9.0 / 5.0,
)


def _normalize_aspect_ratio_candidates(candidates):
    normalized = []
    for aspect_ratio in candidates:
        aspect_ratio = float(aspect_ratio)
        if aspect_ratio <= 0:
            raise ValueError(
                f"aspect ratio candidates must be positive, got {aspect_ratio}"
            )
        if aspect_ratio not in normalized:
            normalized.append(aspect_ratio)
    return tuple(normalized)


class DynamicTorchDataset:
    def __init__(
        self,
        dataset: dict,
        common_config: dict,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool = True,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        seed: int = 42,
        per_gpu_batch_scale: float = 1.0,
        use_spawn: bool = False,
    ) -> None:
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        self.seed = seed
        self.use_spawn = use_spawn and num_workers > 0

        self.dataset = instantiate(
            dataset,
            common_config=common_config,
            _recursive_=False,
        )
        aspect_ratio_range = common_config.augs.aspects
        image_num_range = common_config.frames_per_sample_range
        if (
            len(aspect_ratio_range) != 2
            or aspect_ratio_range[0] > aspect_ratio_range[1]
        ):
            raise ValueError(
                "aspect_ratio_range must be [min, max] with min <= max, "
                f"got {aspect_ratio_range}"
            )
        if (
            len(image_num_range) != 2
            or image_num_range[0] < 1
            or image_num_range[0] > image_num_range[1]
        ):
            raise ValueError(
                "image_num_range must be [min, max] with 1 <= min <= max, "
                f"got {image_num_range}"
            )

        aspect_ratio_candidates = None
        if getattr(common_config.augs, "use_discrete_aspect_ratios", False):
            candidates = getattr(
                common_config.augs,
                "aspect_ratio_candidates",
                DEFAULT_DISCRETE_ASPECT_RATIOS,
            )
            aspect_ratio_candidates = tuple(
                ratio
                for ratio in _normalize_aspect_ratio_candidates(candidates)
                if aspect_ratio_range[0] <= ratio <= aspect_ratio_range[1]
            )
            if not aspect_ratio_candidates:
                raise ValueError(
                    "No discrete aspect ratios remain after applying "
                    f"aspect_ratio_range {aspect_ratio_range}"
                )
            logging.info(
                "Using discrete aspect ratios: %s",
                [round(ratio, 4) for ratio in aspect_ratio_candidates],
            )

        self.sampler = DynamicDistributedSampler(
            self.dataset,
            seed=seed,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        self.batch_sampler = DynamicBatchSampler(
            self.sampler,
            aspect_ratio_range,
            image_num_range,
            aspect_ratio_candidates=aspect_ratio_candidates,
            seed=seed,
            per_gpu_batch_scale=per_gpu_batch_scale,
        )

    def get_loader(self, epoch):
        self.batch_sampler.set_epoch(epoch)
        logging.info("Building dynamic dataloader for epoch %s", epoch)

        loader_kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "batch_sampler": self.batch_sampler,
            "collate_fn": self.collate_fn,
            "worker_init_fn": get_worker_init_fn(
                seed=self.seed,
                epoch=epoch,
                worker_init_fn=self.worker_init_fn,
            ),
        }
        if self.use_spawn:
            loader_kwargs["multiprocessing_context"] = "spawn"
        return DataLoader(self.dataset, **loader_kwargs)


class DynamicBatchSampler(Sampler):
    def __init__(
        self,
        sampler,
        aspect_ratio_range,
        image_num_range,
        aspect_ratio_candidates=None,
        epoch=0,
        per_gpu_batch_scale=1.0,
        seed=42,
    ):
        if per_gpu_batch_scale <= 0:
            raise ValueError(
                f"per_gpu_batch_scale must be positive, got {per_gpu_batch_scale}"
            )

        self.sampler = sampler
        self.aspect_ratio_range = aspect_ratio_range
        self.aspect_ratio_candidates = aspect_ratio_candidates
        self.rng = random.Random()
        self.per_gpu_batch_scale = per_gpu_batch_scale
        self.seed = seed

        image_num_weights = FRAME_COUNT_SAMPLING_WEIGHTS
        self.img_num_to_batch_size = FRAME_COUNT_TO_BATCH_SIZE
        requested_nums = set(range(image_num_range[0], image_num_range[1] + 1))
        missing_weights = sorted(requested_nums - image_num_weights.keys())
        missing_batch_sizes = sorted(requested_nums - self.img_num_to_batch_size.keys())
        if missing_weights or missing_batch_sizes:
            raise ValueError(
                "Frame range is not covered by sampler presets: "
                f"missing weights={missing_weights}, "
                f"missing batch sizes={missing_batch_sizes}"
            )

        self.possible_nums = np.array(sorted(requested_nums))
        weights = np.array(
            [image_num_weights[number] for number in self.possible_nums],
            dtype=np.float64,
        )
        self.normalized_weights = weights / weights.sum()
        self.set_epoch(epoch)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
        seed_bytes = hashlib.sha256(
            f"{self.seed}_{epoch}_{self.sampler.rank}".encode("utf-8")
        ).digest()
        self.rng.seed(int.from_bytes(seed_bytes[:4], byteorder="big"))

    def __iter__(self):
        sampler_iterator = iter(self.sampler)
        while True:
            image_num = int(
                np.random.choice(
                    self.possible_nums,
                    p=self.normalized_weights,
                )
            )
            if self.aspect_ratio_candidates is not None:
                aspect_ratio = self.rng.choice(self.aspect_ratio_candidates)
            else:
                aspect_ratio = round(
                    self.rng.uniform(*self.aspect_ratio_range),
                    2,
                )
                aspect_ratio = 1.0 / aspect_ratio

            self.sampler.update_parameters(
                aspect_ratio=aspect_ratio,
                image_num=image_num,
            )
            batch_size = max(
                1,
                int(
                    np.floor(
                        self.img_num_to_batch_size[image_num] * self.per_gpu_batch_scale
                    )
                ),
            )

            current_batch = []
            for _ in range(batch_size):
                try:
                    current_batch.append(next(sampler_iterator))
                except StopIteration:
                    break
            if not current_batch:
                return
            yield current_batch

    def __len__(self):
        return 1_000_000


class DynamicDistributedSampler(DistributedSampler):
    """Attach the current frame count and aspect ratio to sampled indices."""

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = True,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.aspect_ratio = None
        self.image_num = None

    def __iter__(self):
        for index in super().__iter__():
            yield index, self.image_num, self.aspect_ratio

    def update_parameters(self, aspect_ratio, image_num):
        self.aspect_ratio = aspect_ratio
        self.image_num = image_num
