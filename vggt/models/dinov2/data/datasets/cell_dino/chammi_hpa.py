# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC licence,
# found in the LICENSE_CELL_DINO_CODE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from typing import Any, Callable, Optional, Union

import numpy as np

from ..extended import ExtendedVisionDataset
from ..decoders import DecoderType

logger = logging.getLogger("dinov2")


METADATA_FILE = "morphem70k_v2.csv"

CLASS_LABELS = {
    "golgi apparatus": 0,
    "microtubules": 1,
    "mitochondria": 2,
    "nuclear speckles": 3,
    "cytosol": 4,  # labels only seen in TASK_THREE
    "endoplasmic reticulum": 5,
    "nucleoplasm": 6,
}


class _Split(Enum):
    TRAIN = "Train"
    TASK_ONE = "Task_one"
    TASK_TWO = "Task_two"
    TASK_THREE = "Task_three"


def _load_file_names_and_targets(
    root: str,
    split: _Split,
):
    image_paths = []
    labels = []
    with open(os.path.join(root, METADATA_FILE)) as metadata:
        metadata_reader = csv.DictReader(metadata)
        for row in metadata_reader:
            row_dataset = row["file_path"].split("/")[0]
            if row["train_test_split"].upper() == split and row_dataset == "HPA":
                image_paths.append(row["file_path"])
                labels.append(CLASS_LABELS[row["label"]])

    return image_paths, labels


class CHAMMI_HPA(ExtendedVisionDataset):
    """
    Implementation of the CP (Cell-Painting) subset of the CHAMMI benchmark dataset,
    following the CHAMMI paper: https://arxiv.org/pdf/2310.19224
    Github code: https://github.com/chaudatascience/channel_adaptive_models
    """

    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "CHAMMI_HPA.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_decoder_type: DecoderType = DecoderType.XChannelsDecoder,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            root,
            transforms,
            transform,
            target_transform,
            image_decoder_type=image_decoder_type,
            **kwargs,
        )
        self.split = split
        self.root = root
        self.num_additional_labels_loo_eval = 3

        self._image_paths, self._targets = _load_file_names_and_targets(
            root,
            split,
        )

    def get_image_relpath(self, index: int) -> str:
        return self._image_paths[index]

    def get_image_data(self, index: int) -> bytes:
        image_relpath = self.get_image_relpath(index)
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Any:
        return self._targets[index]

    def get_targets(self) -> np.ndarray:
        return np.array(self._targets)

    def __len__(self) -> int:
        return len(self._image_paths)
