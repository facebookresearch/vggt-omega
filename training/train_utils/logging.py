# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys

from .general import makedir


def setup_logging(
    output_dir=None,
    rank=0,
    log_level_primary="INFO",
    log_level_secondary="ERROR",
    all_ranks: bool = False,
):
    """Send logs to stdout on every rank, and to a file on rank 0 (or all ranks if asked)."""
    log_filename = None
    if output_dir:
        makedir(output_dir)
        if rank == 0:
            log_filename = f"{output_dir}/log.txt"
        elif all_ranks:
            log_filename = f"{output_dir}/log_{rank}.txt"

    logger = logging.getLogger()
    logger.setLevel(log_level_primary)

    formatter = logging.Formatter("%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s")

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level_primary if rank == 0 else log_level_secondary)
    logger.addHandler(console_handler)

    if log_filename is not None:
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level_primary)
        logger.addHandler(file_handler)
