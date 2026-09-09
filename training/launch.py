# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from hydra import compose, initialize
from omegaconf import OmegaConf

from trainer import Trainer


def load_config(config_name: str = "default", overrides=None):
    """Compose the config exactly as training does, so callers can inspect it without training."""
    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=config_name, overrides=list(overrides or []))

    # Resolve ${...} in place but keep a DictConfig: the trainer reads nested config with
    # attribute access, which a plain dict does not support.
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    # exp_name only exists so other paths can interpolate ${exp_name}; it is not a Trainer argument.
    cfg.pop("exp_name", None)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Train VGGT-Omega")
    parser.add_argument("--config", default="default", help="config name under training/config, without .yaml")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="hydra-style overrides, e.g. max_epochs=1 optim.optimizer.lr=1e-5",
    )
    args = parser.parse_args()

    Trainer(**load_config(args.config, args.overrides)).run()


if __name__ == "__main__":
    main()
