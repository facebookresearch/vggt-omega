# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


FRAME_COUNT_TO_BATCH_SIZE = {
    1: 24,
    2: 18,
    3: 12,
    4: 10,
    5: 8,
    6: 6,
    7: 6,
    8: 4,
    9: 4,
    10: 3,
    11: 3,
    12: 3,
    **{frame_count: 2 for frame_count in range(13, 33)},
    **{frame_count: 1 for frame_count in range(33, 106)},
}

FRAME_COUNT_SAMPLING_WEIGHTS = {
    frame_count: (1.0 if frame_count <= 15 else 0.5 if frame_count <= 24 else 0.2)
    for frame_count in range(1, 105)
}
FRAME_COUNT_SAMPLING_WEIGHTS[1] = 0.1
