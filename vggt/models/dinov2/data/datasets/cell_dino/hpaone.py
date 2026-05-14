# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC licence,
# found in the LICENSE_CELL_DINO_CODE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

from ..extended import ExtendedVisionDataset
from ..decoders import DecoderType

logger = logging.getLogger("dinov2")

PROTEIN_LOCALIZATION = [
    "actin filaments,focal adhesion sites",
    "aggresome",
    "centrosome,centriolar satellite",
    "cytosol",
    "endoplasmic reticulum",
    "golgi apparatus",
    "intermediate filaments",
    "microtubules",
    "mitochondria",
    "mitotic spindle",
    "no staining",
    "nuclear bodies",
    "nuclear membrane",
    "nuclear speckles",
    "nucleoli",
    "nucleoli fibrillar center",
    "nucleoplasm",
    "plasma membrane,cell junctions",
    "vesicles,peroxisomes,endosomes,lysosomes,lipid droplets,cytoplasmic bodies",
]  # 19


CELL_TYPE = [
    "A-431",  # 0
    "A549",
    "AF22",
    "ASC TERT1",
    "BJ",
    "CACO-2",
    "EFO-21",
    "HAP1",
    "HDLM-2",
    "HEK 293",  # 9
    "HEL",
    "HUVEC TERT2",
    "HaCaT",
    "HeLa",
    "Hep G2",
    "JURKAT",
    "K-562",
    "MCF7",
    "PC-3",
    "REH",
    "RH-30",  # 20
    "RPTEC TERT1",
    "RT4",
    "SH-SY5Y",
    "SK-MEL-30",
    "SiHa",
    "U-2 OS",
    "U-251 MG",
    "hTCEpi",  # 28
]  # 29 cell types


class _Split(Enum):
    VAL = "val"
    TRAIN = "train"
    ALL = "all"  # images without labels, for encoder training


class _Mode(Enum):
    PROTEIN_LOCALIZATION = "protein_localization"
    CELL_TYPE = "cell_type"

    @property
    def num_labels(self):
        if self == _Mode.CELL_TYPE.value.upper():
            return len(CELL_TYPE)
        return len(PROTEIN_LOCALIZATION)


def _simple_parse_csv(img_rootdir, csv_filepath: str):
    samples = []
    with open(csv_filepath) as filename:
        template = csv.DictReader(filename)
        samples = [(os.path.join(img_rootdir, row["img_path"]), 0) for row in template]
    return samples


def _parse_csv(img_rootdir, csv_labels_path: str):
    nb_protein_location = len(PROTEIN_LOCALIZATION)
    nb_cell_type = len(CELL_TYPE)
    samples = []
    with open(csv_labels_path) as filename:
        reader = csv.DictReader(filename)
        for row in reader:
            protein_location = np.zeros(nb_protein_location, dtype=np.int_)
            for k in range(nb_protein_location):
                if row[PROTEIN_LOCALIZATION[k]] == "True":
                    protein_location[k] = 1

            cell_type = 0
            for k in range(nb_cell_type):
                if row[CELL_TYPE[k]] == "True":
                    cell_type = k

            samples.append(
                (
                    img_rootdir + "/" + row["file"].rsplit("/", 1)[1],
                    protein_location,
                    cell_type,
                )
            )
    return samples


def _load_file_names_and_labels_ssl(
    root: str,
) -> Tuple[List[str], List[Any]]:
    curr_dir_train = os.path.join(root, "varied_size_masked_single_cells_HPA")
    csv_all_path = os.path.join(root, "varied_size_masked_single_cells_pretrain_20240507.csv")
    samples = _simple_parse_csv(curr_dir_train, csv_all_path)
    image_paths, fake_labels = zip(*samples)
    lab = list(fake_labels)
    return image_paths, lab


def _load_file_names_and_labels_train_or_test(
    root: str,
    split: _Split,
    mode: _Mode,
) -> Tuple[List[str], List[Any]]:

    if split == _Split.TRAIN.value.upper() or split == _Split.TRAIN:
        csv_labels_path = os.path.join(root, "fixed_size_masked_single_cells_pretrain_20240507.csv")
    elif split == _Split.VAL.value.upper() or split == _Split.VAL:
        csv_labels_path = os.path.join(root, "fixed_size_masked_single_cells_evaluation_20240507.csv")
    else:
        print("wrong split name")
    curr_dir_val = os.path.join(root, "fixed_size_masked_single_cells_HPA")

    samples = _parse_csv(curr_dir_val, csv_labels_path)
    image_paths, protein_location, cell_type = zip(*samples)
    if mode == _Mode.PROTEIN_LOCALIZATION.value.upper():
        lab = protein_location
    elif mode == _Mode.CELL_TYPE.value.upper():
        lab = cell_type
    else:
        lab = protein_location, cell_type
    image_paths = list(image_paths)
    return image_paths, lab


class HPAone(ExtendedVisionDataset):
    Split = Union[_Split]
    Mode = Union[_Mode]

    def __init__(
        self,
        *,
        split: "HPAone.Split" = _Split.ALL,
        mode: "HPAone.Mode" = None,
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
        self.mode = mode
        self.split = split
        self.root = root

        if (
            split in {_Split.TRAIN.value.upper(), _Split.VAL.value.upper()}
            or split == _Split.TRAIN
            or split == _Split.VAL
        ):
            (
                self._image_paths,
                self._labels,
            ) = _load_file_names_and_labels_train_or_test(root, split, mode)
        elif split == _Split.ALL.value.upper() or split == _Split.ALL:
            self._image_paths, self._labels = _load_file_names_and_labels_ssl(root)
        else:
            logger.info(f"unknown split: {split}, {_Split.ALL.value.upper()}")

    def get_image_relpath(self, index: int) -> str:
        return self._image_paths[index]

    def get_image_data(self, index: int) -> bytes:
        image_relpath = self.get_image_relpath(index)
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Any:
        return self._labels[index]

    def get_targets(self) -> np.ndarray:
        return np.array(self._labels)

    def __len__(self) -> int:
        return len(self._image_paths)
