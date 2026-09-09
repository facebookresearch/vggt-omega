# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Turn an FSDP (DCP) sharded checkpoint directory into a single .pt file.

Training with strategy: fsdp writes each checkpoint as a directory of shards plus a
metadata.pt. The inference code loads a single state dict, so convert first:

    python consolidate_checkpoint.py logs/exp/checkpoints/checkpoint_5 checkpoint_5.pt

Only model weights are extracted; optimizer shards are skipped, which keeps peak memory to
roughly the model size.
"""

import argparse
import os

import torch
import torch.distributed.checkpoint as dcp


def consolidate(input_dir: str, output_file: str, wrap_key: str = "model") -> None:
    reader = dcp.FileSystemReader(input_dir)
    metadata = reader.read_metadata()

    entries = metadata.state_dict_metadata
    has_model_prefix = any(k.startswith("model.") for k in entries)

    target = {}
    for key, meta in entries.items():
        if not isinstance(meta, dcp.TensorStorageMetadata):
            continue
        if has_model_prefix:
            if not key.startswith("model."):
                continue
        elif "optimizer" in key:
            continue
        target[key] = torch.empty(meta.size, dtype=meta.properties.dtype, device="cpu")

    if not target:
        raise ValueError(f"No model tensors found in {input_dir}")
    print(f"Loading {len(target)} tensors from {input_dir}")
    dcp.load(target, storage_reader=reader)

    if has_model_prefix:
        target = {(k[len("model.") :] if k.startswith("model.") else k): v for k, v in target.items()}

    torch.save({wrap_key: target} if wrap_key else target, output_file)
    print(f"Wrote {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate an FSDP/DCP checkpoint into one .pt")
    parser.add_argument("input_dir", help="Checkpoint directory containing .metadata")
    parser.add_argument("output_file", help="Destination .pt path")
    parser.add_argument(
        "--wrap-key",
        default="model",
        help="Key to nest the state dict under. Pass an empty string for a bare state dict.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"{args.input_dir} is not a directory")
    consolidate(args.input_dir, args.output_file, args.wrap_key)


if __name__ == "__main__":
    main()
