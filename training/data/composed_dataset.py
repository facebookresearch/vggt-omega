# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import random

import numpy as np
import torch
from hydra.utils import instantiate
from torch.utils.data import ConcatDataset, Dataset

from train_utils.normalization import normalize_camera_extrinsics_and_points_batch

from .augmentation import get_image_augmentation
from .dataset_util import apply_random_patch_mask
from .track_util import build_tracks_by_depth


class ComposedDataset(Dataset):
    """Combine weighted base datasets and apply shared sample processing."""

    def __init__(self, dataset_configs, common_config):
        configs = (
            dataset_configs.values()
            if hasattr(dataset_configs, "values")
            else dataset_configs
        )
        datasets = [
            instantiate(config, common_conf=common_config) for config in configs
        ]
        self.base_dataset = TupleConcatDataset(datasets, common_config)

        self.cojitter = common_config.augs.cojitter
        self.cojitter_ratio = common_config.augs.cojitter_ratio
        if getattr(common_config.augs, "enable_image_aug", True):
            self.image_aug = get_image_augmentation(
                color_jitter=common_config.augs.color_jitter,
                gray_scale=common_config.augs.gray_scale,
                gau_blur=common_config.augs.gau_blur,
                jpeg_compression=getattr(common_config.augs, "jpeg_compression", False),
                low_res_blur=getattr(common_config.augs, "low_res_blur", -1),
                low_res_blur_min_ratio=getattr(
                    common_config.augs,
                    "low_res_blur_min_ratio",
                    0.25,
                ),
            )
        else:
            self.image_aug = None

        self.fixed_num_images = common_config.fix_img_num
        self.fixed_aspect_ratio = common_config.fix_aspect_ratio
        self.load_track = common_config.load_track
        self.track_num = common_config.track_num
        self.training = common_config.training
        self.random_patch_mask_ratio = getattr(
            common_config,
            "random_patch_mask_ratio",
            -1,
        )
        self.random_patch_mask_size = getattr(
            common_config,
            "random_patch_mask_size",
            -1,
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx_tuple):
        if self.fixed_num_images > 0:
            seq_idx = idx_tuple[0] if isinstance(idx_tuple, tuple) else idx_tuple
            idx_tuple = (
                seq_idx,
                self.fixed_num_images,
                self.fixed_aspect_ratio,
            )

        batch = self.base_dataset[idx_tuple]
        images = torch.from_numpy(np.stack(batch["images"]))
        images = (
            images.permute(0, 3, 1, 2).to(dtype=torch.get_default_dtype()).div_(255)
        )
        depths = torch.from_numpy(np.stack(batch["depths"]).astype(np.float32))
        extrinsics = torch.from_numpy(np.stack(batch["extrinsics"]).astype(np.float32))
        intrinsics = torch.from_numpy(np.stack(batch["intrinsics"]).astype(np.float32))
        world_points = torch.from_numpy(
            np.stack(batch["world_points"]).astype(np.float32)
        )

        if self.training and self.image_aug is not None:
            if self.cojitter and random.random() > self.cojitter_ratio:
                images = self.image_aug(images)
            else:
                for image_idx in range(len(images)):
                    images[image_idx] = self.image_aug(images[image_idx])

        if self.random_patch_mask_ratio > 0 and self.random_patch_mask_size > 0:
            images = images.clone()
            depths = depths.clone()
            for _ in range(3):
                selected = torch.rand(len(images)) < self.random_patch_mask_ratio
                if selected.any():
                    images[selected], depths[selected] = apply_random_patch_mask(
                        images[selected],
                        depths[selected],
                        self.random_patch_mask_size,
                    )

        point_masks = depths > 1e-3
        sample = {
            "dataset": batch["dataset"],
            "seq_name": batch["seq_name"],
            "is_synthetic": batch.get("is_synthetic", False),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "world_points": world_points,
            "point_masks": point_masks,
            "depth_train_mask": batch.get("depth_train_mask", True),
        }

        if self.load_track:
            tracks, track_vis_mask = build_tracks_by_depth(
                extrinsics,
                intrinsics,
                world_points,
                depths,
                point_masks,
                target_track_num=self.track_num,
                seq_name=batch["seq_name"],
            )
            sample["tracks"] = tracks
            sample["track_vis_mask"] = track_vis_mask

        return normalize_camera_extrinsics_and_points_batch(sample)


class TupleConcatDataset(ConcatDataset):
    """Pass dynamic tuple indices through a weighted concatenation."""

    def __init__(self, datasets, common_config):
        super().__init__(datasets)
        self.inside_random = common_config.inside_random

    def __getitem__(self, idx):
        if not isinstance(idx, tuple) or len(idx) != 3:
            raise ValueError(
                "Index must be a tuple of (sample_idx, num_images, aspect_ratio)"
            )

        sample_idx, num_images, aspect_ratio = idx
        if self.inside_random:
            sample_idx = random.randrange(self.cumulative_sizes[-1])

        if sample_idx < 0:
            if -sample_idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            sample_idx += len(self)

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, sample_idx)
        dataset_offset = (
            0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
        )
        local_idx = sample_idx - dataset_offset
        return self.datasets[dataset_idx][(local_idx, num_images, aspect_ratio)]
