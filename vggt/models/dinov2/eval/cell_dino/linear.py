# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC licence,
# found in the LICENSE_CELL_DINO_CODE file in the root directory of this source tree.

import argparse
from functools import partial
import json
import logging
import os
import sys
from typing import Any, Callable, Dict, Optional, Tuple, List
from enum import Enum
from dataclasses import dataclass

from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.nn.parallel import DistributedDataParallel


from dinov2.data import SamplerType, make_data_loader, make_dataset, DatasetWithEnumeratedTargets
from dinov2.data.cell_dino.transforms import NormalizationType, make_classification_eval_cell_transform
import dinov2.distributed as distributed
from dinov2.eval.metrics import MetricType, build_metric
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model
from dinov2.eval.cell_dino.utils import (
    evaluate_with_accumulate,
    LossType,
    average_metrics,
    create_train_dataset_dict,
    get_num_classes,
    extract_features_for_dataset_dict,
)
from dinov2.eval.utils import ModelWithIntermediateLayers
from dinov2.logging import MetricLogger
from dinov2.utils.checkpoint import build_periodic_checkpointer, resume_or_load

logger = logging.getLogger("dinov2")

"""
List of changes with respect to the standard linear evaluation script:

bag of channel option : SCALE ADAPTIVE STRATEGY

Adam optimizer instead of SGD
Scheduler : two options : onecycleLR or CosineAnnealingLR
the transforms/normalization are different, now calling make_classification_eval_cell_transform
add binary cross entropy loss option for protein localization
change the definition of the num_classes using get_num_classes
change of some default parameters (batch_size, epoch_length, epochs, lrs)
defined n_last_blocks option
avgpool option
leave one out strategy for CHAMMI evaluation
grid search for optimal weight decay
"""


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parents = parents or []
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--train-dataset",
        dest="train_dataset_str",
        type=str,
        help="Training dataset",
    )
    parser.add_argument(
        "--val-dataset",
        dest="val_dataset_str",
        type=str,
        help="Validation dataset",
    )
    parser.add_argument(
        "--test-datasets",
        dest="test_dataset_strs",
        type=str,
        nargs="+",
        help="Test datasets, none to reuse the validation dataset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch Size (per GPU)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number de Workers",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        help="Length of an epoch in number of iterations",
    )
    parser.add_argument(
        "--save-checkpoint-frequency",
        type=int,
        help="Number of epochs between two named checkpoint saves.",
    )
    parser.add_argument(
        "--eval-period-iterations",
        type=int,
        help="Number of iterations between two evaluations.",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        help="Learning rates to grid search.",
    )
    parser.add_argument(
        "--weight_decays",
        nargs="+",
        type=float,
        help="Weight decays to grid search.",
    )
    parser.add_argument(
        "--n-last-blocks",
        type=int,
        help="number of backbone last blocks used for the linear classifier",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not resume from existing checkpoints",
    )
    parser.add_argument(
        "--val-metric-type",
        type=MetricType,
        choices=list(MetricType),
        help="Validation metric",
    )
    parser.add_argument(
        "--test-metric-types",
        type=MetricType,
        choices=list(MetricType),
        nargs="+",
        help="Evaluation metric",
    )
    parser.add_argument(
        "--classifier-fpath",
        type=str,
        help="Path to a file containing pretrained linear classifiers",
    )
    parser.add_argument(
        "--val-class-mapping-fpath",
        type=str,
        help="Path to a file containing a mapping to adjust classifier outputs",
    )
    parser.add_argument(
        "--test-class-mapping-fpaths",
        nargs="+",
        type=str,
        help="Path to a file containing a mapping to adjust classifier outputs",
    )
    parser.add_argument(
        "--loss-type",
        type=LossType,
        help="Cross Entropy or Binary Cross Entropy, default cross entropy loss",
    )
    parser.add_argument(
        "--bag-of-channels",
        action="store_true",
        help='Whether to use the "bag of channels" channel adaptive strategy',
    )
    parser.add_argument(
        "--leave-one-out-dataset",
        type=str,
        help="Path with indexes to use the leave one out strategy for CHAMMI_CP task 3 and CHAMMI_HPA task 4",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        help="crop size for train and eval",
    )
    parser.add_argument(
        "--resize-size",
        type=int,
        help="resize size for image just before crop. 0: no resize",
    )
    parser.add_argument(
        "--avgpool",
        action="store_true",
        help="Whether to use average pooling of path tokens in addition to CLS tokens",
    )
    parser.add_argument(
        "--scheduler",
        type=SchedulerType,
        help="Scheduler type",
    )

    parser.set_defaults(
        train_dataset_str="ImageNet:split=TRAIN",
        val_dataset_str="ImageNet:split=VAL",
        test_dataset_strs=None,
        epochs=30,
        batch_size=64,
        num_workers=8,
        epoch_length=145,
        save_checkpoint_frequency=1250,
        eval_period_iterations=1250,
        learning_rates=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0],
        weight_decays=[0.0, 0.0001, 1.0e-05],
        val_metric_type=MetricType.MEAN_ACCURACY,
        test_metric_types=None,
        classifier_fpath=None,
        val_class_mapping_fpath=None,
        test_class_mapping_fpaths=[None],
        loss_type=LossType.CROSS_ENTROPY,
        crop_size=384,
        resize_size=0,
        n_last_blocks=4,
        avgpool=False,
        scheduler=SchedulerType.COSINE_ANNEALING,
    )
    return parser


def has_ddp_wrapper(m: nn.Module) -> bool:
    return isinstance(m, DistributedDataParallel)


def remove_ddp_wrapper(m: nn.Module) -> nn.Module:
    return m.module if has_ddp_wrapper(m) else m


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool, bag_of_channels):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if bag_of_channels:
        if use_avgpool:
            output = torch.cat(
                (
                    output,
                    torch.mean(intermediate_output[-1][0], dim=-2).reshape(intermediate_output[-1][0].shape[0], -1),
                    # average pooling of patch tokens: average over N, then concatenate channels if single-channel patch model
                ),
                dim=-1,
            )  # concatenate average pooling of patch tokens to concatenated patch tokens
    else:
        if use_avgpool:
            output = torch.cat(
                (
                    output,
                    torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
                ),
                dim=-1,
            )
    output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(
        self, out_dim, use_n_blocks, use_avgpool, num_classes=1000, bag_of_channels=False, leave_one_out=False
    ):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.bag_of_channels = bag_of_channels
        self.leave_one_out = leave_one_out
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        if self.leave_one_out:
            return self.linear(x_tokens_list)
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool, self.bag_of_channels)
        return self.linear(output)


class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


class LinearPostprocessor(nn.Module):
    def __init__(self, linear_classifier, class_mapping=None):
        super().__init__()
        self.linear_classifier = linear_classifier
        self.register_buffer("class_mapping", None if class_mapping is None else torch.LongTensor(class_mapping))

    def forward(self, samples, targets):
        preds = self.linear_classifier(samples)
        return {
            "preds": preds[:, self.class_mapping] if self.class_mapping is not None else preds,
            "target": targets,
        }


def scale_lr(learning_rates, batch_size):
    return learning_rates * (batch_size * distributed.get_global_size()) / 256.0


def setup_linear_classifiers(
    sample_output,
    n_last_blocks_list,
    learning_rates,
    weight_decays,
    batch_size,
    num_classes=1000,
    bag_of_channels=False,
    leave_one_out=False,
    avgpool=False,
):
    linear_classifiers_dict = nn.ModuleDict()
    avgpool_value = avgpool
    optim_param_groups = []
    for n in n_last_blocks_list:
        for avgpool in [avgpool_value]:
            for _lr in learning_rates:
                for wd in weight_decays:
                    lr = scale_lr(_lr, batch_size)
                    out_dim = create_linear_input(
                        sample_output, use_n_blocks=n, use_avgpool=avgpool, bag_of_channels=bag_of_channels
                    ).shape[1]
                    linear_classifier = LinearClassifier(
                        out_dim,
                        use_n_blocks=n,
                        use_avgpool=avgpool,
                        num_classes=num_classes,
                        bag_of_channels=bag_of_channels,
                        leave_one_out=leave_one_out,
                    )
                    linear_classifier = linear_classifier.cuda()
                    linear_classifiers_dict[
                        f"classifier_{n}_blocks_avgpool_{avgpool}_lr_{lr:.5f}_wd_{wd:.2E}".replace(".", "_")
                    ] = linear_classifier
                    optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr, "weight_decay": wd})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)
    if distributed.is_enabled():
        linear_classifiers = nn.parallel.DistributedDataParallel(linear_classifiers)

    return linear_classifiers, optim_param_groups


def make_eval_data_loader(
    *,
    test_dataset_str_or_path_or_loo_dataset,
    config,
    batch_size,
    num_workers,
):
    if isinstance(test_dataset_str_or_path_or_loo_dataset, str):
        logger.info(f"Loading dataset {test_dataset_str_or_path_or_loo_dataset}")
        transform = make_classification_eval_cell_transform(
            normalization_type=NormalizationType.SELF_NORM_CENTER_CROP,
            resize_size=config["resize_size"],
            crop_size=config["crop_size"],
        )
        test_dataset = make_dataset(dataset_str=test_dataset_str_or_path_or_loo_dataset, transform=transform)
        collate_fn = None
    else:
        logger.info("Making data loader for feature dataset (typical in leave one out evaluation)")
        test_dataset = test_dataset_str_or_path_or_loo_dataset
        collate_fn = None
    class_mapping = None
    if hasattr(test_dataset, "get_imagenet_class_mapping"):
        class_mapping = test_dataset.get_imagenet_class_mapping()

    test_data_loader = make_data_loader(
        dataset=DatasetWithEnumeratedTargets(
            test_dataset, pad_dataset=True, num_replicas=distributed.get_global_size()
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=False,
        collate_fn=collate_fn,
    )
    return test_data_loader, class_mapping


@dataclass
class Evaluator:
    batch_size: int
    num_workers: int
    dataset_str_or_path: str
    config: Dict
    metric_type: MetricType
    metrics_file_path: str
    training_num_classes: int
    save_results_func: Optional[Callable]
    val_dataset_loo: Optional[TensorDataset] = None

    def __post_init__(self):
        self.main_metric_name = f"{self.dataset_str_or_path}_accuracy"

        if self.val_dataset_loo is not None:
            self.dataset_str_or_path = self.val_dataset_loo

        self.data_loader, self.class_mapping = make_eval_data_loader(
            test_dataset_str_or_path_or_loo_dataset=self.dataset_str_or_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            config=self.config,
        )

    @torch.no_grad()
    def _evaluate_linear_classifiers(
        self,
        *,
        feature_model,
        linear_classifiers,
        iteration,
        prefixstring="",
        best_classifier_on_val=None,
        accumulate_results=False,
        test_mode=False,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, torch.Tensor]]]:
        logger.info("running validation !")

        num_classes = len(self.class_mapping) if self.class_mapping is not None else self.training_num_classes
        metric = build_metric(self.metric_type, num_classes=num_classes)
        postprocessors = {
            k: LinearPostprocessor(v, self.class_mapping) for k, v in linear_classifiers.classifiers_dict.items()
        }
        metrics = {k: metric.clone() for k in linear_classifiers.classifiers_dict}

        _, results_dict_temp, accumulated_results = evaluate_with_accumulate(
            feature_model,
            self.data_loader,
            postprocessors,
            metrics,
            torch.cuda.current_device(),
            accumulate_results=accumulate_results,
            leave_one_out=self.config["leave_one_out"],
            test_mode=test_mode,
        )

        logger.info("")
        results_dict = {}
        max_accuracy = 0
        best_classifier = ""
        for _, (classifier_string, metric) in enumerate(results_dict_temp.items()):
            logger.info(f"{prefixstring} -- Classifier: {classifier_string} * {metric}")
            if (
                best_classifier_on_val is None and metric["top-1"].item() > max_accuracy
            ) or classifier_string == best_classifier_on_val:
                max_accuracy = metric["top-1"].item()
                best_classifier = classifier_string

        results_dict["best_classifier"] = {"name": best_classifier, "accuracy": max_accuracy}

        logger.info(f"best classifier: {results_dict['best_classifier']}")

        accumulated_best_results = None
        if test_mode:
            accumulated_best_results = accumulated_results
        elif accumulated_results is not None:
            accumulated_best_results = accumulated_results[best_classifier]

        if distributed.is_main_process():
            with open(self.metrics_file_path, "a") as f:
                f.write(f"iter: {iteration}\n")
                for k, v in results_dict.items():
                    f.write(json.dumps({k: v}) + "\n")
                f.write("\n")

        return results_dict, accumulated_best_results

    def evaluate_and_maybe_save(
        self,
        feature_model,
        linear_classifiers,
        iteration: int,
        best_classifier_on_val: Optional[Any] = None,
        save_filename_suffix: str = "",
        prefixstring: str = "",
        test_mode: bool = False,
    ):
        logger.info(f"Testing on {self.dataset_str_or_path}")
        save_results = self.save_results_func is not None
        full_results_dict, accumulated_best_results = self._evaluate_linear_classifiers(
            feature_model=feature_model,
            linear_classifiers=remove_ddp_wrapper(linear_classifiers),
            iteration=iteration,
            prefixstring=prefixstring,
            best_classifier_on_val=best_classifier_on_val,
            accumulate_results=save_results,
            test_mode=test_mode,
        )
        if self.save_results_func is not None:
            self.save_results_func(
                filename_suffix=f"{self.dataset_str_or_path}{save_filename_suffix}", **accumulated_best_results
            )

        results_dict = {
            self.main_metric_name: 100.0 * full_results_dict["best_classifier"]["accuracy"],
            "best_classifier": full_results_dict["best_classifier"]["name"],
        }
        return results_dict, accumulated_best_results


def make_evaluators(
    config: Dict,
    val_metric_type: MetricType,
    val_dataset: str,
    metric_type: MetricType,
    metrics_file_path: str,
    training_num_classes: int,
    save_results_func: Optional[Callable],
    val_dataset_loo: Optional[TensorDataset] = None,
):
    test_metric_types = config["test_metric_types"]
    test_dataset_strs = config["test_datasets"]
    if test_dataset_strs is None:
        test_dataset_strs = (config["val_dataset"],)
    if test_metric_types is None:
        test_metric_types = (val_metric_type,)
    else:
        assert len(test_metric_types) == len(config["test_datasets"])

    val_evaluator, *test_evaluators = [
        Evaluator(
            dataset_str_or_path=dataset_str_or_path,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            config=config,
            metric_type=metric_type,
            metrics_file_path=metrics_file_path,
            training_num_classes=training_num_classes,
            save_results_func=save_results_func,
            val_dataset_loo=val_dataset_loo,
        )
        for dataset_str_or_path, metric_type in zip(
            (val_dataset,) + tuple(test_dataset_strs),
            (val_metric_type,) + tuple(test_metric_types),
        )
    ]
    return val_evaluator, test_evaluators


class SchedulerType(Enum):
    COSINE_ANNEALING = "cosine_annealing"
    ONE_CYCLE = "one_cycle"

    def get_scheduler(self, optimizer, optim_param_groups, epoch_length, epochs, max_iter):
        if self == SchedulerType.ONE_CYCLE:
            lr_list = [optim_param_groups[i]["lr"] for i in range(len(optim_param_groups))]
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr_list, steps_per_epoch=epoch_length, epochs=epochs
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
            print("CosineAnnealingLR scheduler")
        return scheduler


def setup_linear_training(
    *,
    config: Dict,
    sample_output: torch.Tensor,
    training_num_classes: int,
    checkpoint_output_dir: str,
):
    linear_classifiers, optim_param_groups = setup_linear_classifiers(
        sample_output,
        config["n_last_blocks_list"],
        config["learning_rates"],
        config["weight_decays"],
        config["batch_size"],
        training_num_classes,
        config["bag_of_channels"],
        config["leave_one_out"],
        config["avgpool"],
    )
    max_iter = config["epochs"] * config["epoch_length"]
    optimizer = torch.optim.AdamW(optim_param_groups, weight_decay=0)

    scheduler = config["scheduler"].get_scheduler(
        optimizer=optimizer,
        optim_param_groups=optim_param_groups,
        epoch_length=config["epoch_length"],
        epochs=config["epochs"],
        max_iter=max_iter,
    )
    checkpoint_period = config["save_checkpoint_iterations"] or config["epoch_length"]
    periodic_checkpointer = build_periodic_checkpointer(
        linear_classifiers,
        checkpoint_output_dir,
        optimizer=optimizer,
        scheduler=scheduler,
        period=checkpoint_period,
        max_iter=max_iter,
        max_to_keep=None,
    )
    checkpoint = resume_or_load(periodic_checkpointer, config["classifier_fpath"] or "", resume=config["resume"])

    start_iter = checkpoint.get("iteration", -1) + 1
    best_accuracy = checkpoint.get("best_accuracy", -1)

    if config["loss_type"] == LossType.BINARY_CROSS_ENTROPY:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    return (
        linear_classifiers,
        start_iter,
        max_iter,
        criterion,
        optimizer,
        scheduler,
        periodic_checkpointer,
        best_accuracy,
    )


def train_linear_classifiers(
    *,
    feature_model,
    train_dataset,
    train_config: Dict,
    training_num_classes: int,
    val_evaluator: Evaluator,
    checkpoint_output_dir: str,
    sample_output: Optional[torch.Tensor] = None,
):

    if train_config["leave_one_out"]:
        assert sample_output is not None, "sample_output should be passed as argument when using leave_one_out."
    else:
        sample_output = feature_model(train_dataset[0][0].unsqueeze(0).cuda())

    (
        linear_classifiers,
        start_iter,
        max_iter,
        criterion,
        optimizer,
        scheduler,
        periodic_checkpointer,
        best_accuracy,
    ) = setup_linear_training(
        config=train_config,
        sample_output=sample_output,
        training_num_classes=training_num_classes,
        checkpoint_output_dir=checkpoint_output_dir,
    )

    sampler_type = SamplerType.INFINITE
    train_data_loader = make_data_loader(
        dataset=train_dataset,
        batch_size=train_config["batch_size"],
        num_workers=train_config["num_workers"],
        shuffle=True,
        seed=0,
        sampler_type=sampler_type,
        sampler_advance=start_iter,
        drop_last=True,
        persistent_workers=True,
    )
    eval_period = train_config["eval_period_iterations"] or train_config["epoch_length"]
    iteration = start_iter
    logger.info("Starting training from iteration {}".format(start_iter))
    metric_logger = MetricLogger(delimiter="  ")
    header = "Training"

    for data, labels in metric_logger.log_every(
        train_data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        data = data.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        if not train_config["leave_one_out"]:
            in_classifier = feature_model(data)
        else:
            in_classifier = data

        outputs = linear_classifiers(in_classifier)

        if len(labels.shape) > 1:
            labels = labels.float()
        losses = {f"loss_{k}": criterion(v, labels) for k, v in outputs.items()}
        loss = sum(losses.values())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        periodic_checkpointer.step(iteration=iteration, best_accuracy=best_accuracy)

        if eval_period > 0 and (iteration + 1) % eval_period == 0 and iteration != max_iter - 1:
            val_results_dict, _ = val_evaluator.evaluate_and_maybe_save(
                feature_model=feature_model,
                linear_classifiers=linear_classifiers,
                prefixstring=f"ITER: {iteration}",
                iteration=iteration,
            )
            val_accuracy = val_results_dict[val_evaluator.main_metric_name]
            if val_accuracy >= best_accuracy:
                best_accuracy = val_accuracy
                periodic_checkpointer.save_best(iteration=iteration, best_accuracy=best_accuracy)
            torch.distributed.barrier()

        iteration = iteration + 1

    return feature_model, linear_classifiers, iteration, periodic_checkpointer


def eval_linear_with_model(
    model,
    output_dir,
    train_dataset_str,
    val_dataset_str,
    batch_size,
    epochs,
    epoch_length,
    num_workers,
    save_checkpoint_frequency,
    eval_period_iterations,
    learning_rates,
    weight_decays,
    autocast_dtype,
    test_dataset_strs=None,
    resume=True,
    classifier_fpath=None,
    val_metric_type=MetricType.MEAN_ACCURACY,
    test_metric_types=None,
    loss_type=LossType.CROSS_ENTROPY,
    bag_of_channels=False,
    leave_one_out_dataset="",
    resize_size=0,
    crop_size=384,
    n_last_blocks=4,
    avgpool=False,
    scheduler=SchedulerType.COSINE_ANNEALING,
):

    if leave_one_out_dataset == "" or leave_one_out_dataset is None:
        leave_one_out = False
    else:
        logger.info("Reading the leave-one-out label metadata.")

        leave_one_out_indices = {}
        metadata = pd.read_csv(leave_one_out_dataset)
        if "HPA" in leave_one_out_dataset:
            metadata = metadata[metadata["Task_three"]].reset_index()
            leave_one_out_label_type = "cell_type"
        else:
            metadata = metadata[metadata["Task_four"]].reset_index()
            leave_one_out_label_type = "Plate"
        leave_one_out_labels = metadata[leave_one_out_label_type].unique()

        for leave_one_out_label in leave_one_out_labels:
            leave_one_out_indices[leave_one_out_label] = np.array(
                metadata[metadata[leave_one_out_label_type] == leave_one_out_label].index.values
            )

        leave_one_out = True

    train_transform = make_classification_eval_cell_transform(
        normalization_type=NormalizationType.SELF_NORM_AUG_DECODER, crop_size=crop_size, resize_size=resize_size
    )
    print("train_transform", train_transform)
    train_dataset = make_dataset(
        dataset_str=train_dataset_str,
        transform=train_transform,
    )

    training_num_classes = get_num_classes(train_dataset)
    if leave_one_out:
        training_num_classes += train_dataset.num_additional_labels_loo_eval
    train_dataset_dict = create_train_dataset_dict(train_dataset)
    n_last_blocks_list = [n_last_blocks]
    n_last_blocks = max(n_last_blocks_list)
    dataset_use_cache = True
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)

    if bag_of_channels:
        sample = train_dataset[0][0].unsqueeze(0)
        sample_output = feature_model(sample.cuda())

    if leave_one_out:
        loo_dict = {}
        train_data_dict = extract_features_for_dataset_dict(
            feature_model,
            train_dataset_dict,
            batch_size,
            num_workers,
            gather_on_cpu=True,
            avgpool=avgpool,
        )
        val_dataset = make_dataset(
            dataset_str=val_dataset_str,
            transform=make_classification_eval_cell_transform(
                normalization_type=NormalizationType.SELF_NORM_CENTER_CROP, crop_size=crop_size, resize_size=resize_size
            ),
        )
        val_dataset_dict = create_train_dataset_dict(val_dataset)
        val_data_dict = extract_features_for_dataset_dict(
            feature_model,
            val_dataset_dict,
            batch_size,
            num_workers,
            gather_on_cpu=True,
            avgpool=avgpool,
        )

        train_features = train_data_dict[0]["train_features"]
        train_labels = train_data_dict[0]["train_labels"]
        val_features = val_data_dict[0]["train_features"]
        val_labels = val_data_dict[0]["train_labels"]

        for loo_label, loo_indices in leave_one_out_indices.items():
            loo_for_training_indices = torch.ones(val_features.shape[0], dtype=bool)
            loo_for_training_indices[loo_indices] = False
            loo_for_val_indices = torch.zeros(val_features.shape[0], dtype=bool)
            loo_for_val_indices[loo_indices] = True

            loo_dict[loo_label] = {
                "train_features": torch.cat([train_features, val_features[loo_for_training_indices]]),
                "train_labels": torch.cat([train_labels, val_labels[loo_for_training_indices]]),
                "val_features": val_features[loo_indices],
                "val_labels": val_labels[loo_indices],
            }
    save_results_func = None
    # if config.save_results:
    #     save_results_func = partial(default_save_results_func, output_dir=output_dir)

    metrics_file_path = os.path.join(output_dir, "results_eval_linear.json")
    periodic_checkpointers: list = []

    train_config = {
        "learning_rates": learning_rates,
        "weight_decays": weight_decays,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "dataset_use_cache": dataset_use_cache,
        "eval_period_iterations": eval_period_iterations,
        "epoch_length": epoch_length,
        "leave_one_out": leave_one_out,
        "bag_of_channels": bag_of_channels,
        "n_last_blocks_list": n_last_blocks_list,
        "epochs": epochs,
        "loss_type": loss_type,
        "resume": resume,
        "save_checkpoint_iterations": save_checkpoint_frequency,
        "classifier_fpath": classifier_fpath,
        "avgpool": avgpool,
        "scheduler": scheduler,
    }
    config = {
        "test_metric_types": test_metric_types,
        "test_datasets": test_dataset_strs,
        "val_metric_types": val_metric_type,
        "val_dataset": val_dataset_str,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "leave_one_out": leave_one_out,
        "crop_size": crop_size,
        "resize_size": resize_size,
    }
    if not leave_one_out:
        val_evaluator, test_evaluators = make_evaluators(
            config=config,
            val_metric_type=val_metric_type,
            val_dataset=val_dataset_str,
            metric_type=test_metric_types,
            metrics_file_path=metrics_file_path,
            training_num_classes=training_num_classes,
            save_results_func=save_results_func,
        )
        results_dict = {}

        for _try in train_dataset_dict.keys():
            if len(train_dataset_dict) > 1:
                checkpoint_output_dir = os.path.join(output_dir, f"checkpoints_{_try}")
                save_filename_suffix = f"_{_try}"
            else:
                checkpoint_output_dir, save_filename_suffix = output_dir, ""
            os.makedirs(checkpoint_output_dir, exist_ok=True)

            feature_model, linear_classifiers, iteration, periodic_checkpointer = train_linear_classifiers(
                train_config=train_config,
                feature_model=feature_model,
                train_dataset=train_dataset_dict[_try],
                training_num_classes=training_num_classes,
                val_evaluator=val_evaluator,
                checkpoint_output_dir=checkpoint_output_dir,
            )
            periodic_checkpointers.append(periodic_checkpointer)
            results_dict[_try], _ = val_evaluator.evaluate_and_maybe_save(
                feature_model=feature_model,
                linear_classifiers=linear_classifiers,
                iteration=iteration,
                save_filename_suffix=save_filename_suffix,
            )
            for test_evaluator in test_evaluators:
                eval_results_dict, _ = test_evaluator.evaluate_and_maybe_save(
                    feature_model=feature_model,
                    linear_classifiers=linear_classifiers,
                    iteration=iteration,
                    best_classifier_on_val=results_dict[_try]["best_classifier"],
                    save_filename_suffix=save_filename_suffix,
                )
                results_dict[_try] = {**eval_results_dict, **results_dict[_try]}
        if len(train_dataset_dict) > 1:
            results_dict = average_metrics(results_dict, ignore_keys=["best_classifier"])
        else:
            results_dict = {**results_dict[_try]}
    else:  # if leave one out is True
        test_results_dict = {}
        for loo_label in loo_dict.keys():

            checkpoint_output_dir, save_filename_suffix = os.path.join(output_dir, f"checkpoints_{loo_label}"), ""
            os.makedirs(checkpoint_output_dir, exist_ok=True)

            train_dataset_loo = TensorDataset(
                loo_dict[loo_label]["train_features"], loo_dict[loo_label]["train_labels"]
            )

            logger.info(f"Creating leave_one_out evaluators. loo_label: {loo_label}")
            val_dataset_loo = TensorDataset(loo_dict[loo_label]["val_features"], loo_dict[loo_label]["val_labels"])
            val_evaluators_loo, _ = make_evaluators(
                config=config,
                val_metric_type=val_metric_type,
                val_dataset="loo",
                metric_type=test_metric_types,
                metrics_file_path=metrics_file_path,
                training_num_classes=training_num_classes,
                save_results_func=save_results_func,
                val_dataset_loo=val_dataset_loo,
            )
            feature_model, linear_classifiers, iteration, periodic_checkpointer = train_linear_classifiers(
                feature_model=feature_model,
                train_dataset=train_dataset_loo,
                train_config=train_config,
                training_num_classes=training_num_classes,
                val_evaluator=val_evaluators_loo,
                checkpoint_output_dir=checkpoint_output_dir,
                sample_output=sample_output,
            )
            periodic_checkpointers.append(periodic_checkpointer)
            _, test_results_dict[loo_label] = val_evaluators_loo.evaluate_and_maybe_save(
                feature_model=feature_model,
                linear_classifiers=linear_classifiers,
                iteration=iteration,
                save_filename_suffix=save_filename_suffix,
                test_mode=True,
            )
        classifier_names = test_results_dict[loo_label].keys()
        results_dict = {k: [[], []] for k in classifier_names}
        for ll in test_results_dict.keys():
            for k in classifier_names:
                results_dict[k][0].append(test_results_dict[ll][k][0])
                results_dict[k][1].append(test_results_dict[ll][k][1])
        for k in classifier_names:
            results_dict[k] = [
                np.argmax(torch.cat(results_dict[k][0]).cpu().detach().numpy(), axis=1),
                torch.cat(results_dict[k][1]).cpu().detach().numpy(),
            ]
            results_dict[k] = f1_score(results_dict[k][1], results_dict[k][0], average="macro", labels=[4, 5, 6])
        logger.info(
            f"Best performance is for {max(results_dict, key=results_dict.get)}, with F1-Score of {results_dict[max(results_dict, key=results_dict.get)]}"
        )

    logger.info("Test Results Dict " + str(results_dict))
    return results_dict


def main(args):
    model, autocast_dtype = setup_and_build_model(args)
    eval_linear_with_model(
        model=model,
        output_dir=args.output_dir,
        train_dataset_str=args.train_dataset_str,
        val_dataset_str=args.val_dataset_str,
        test_dataset_strs=args.test_dataset_strs,
        batch_size=args.batch_size,
        epochs=args.epochs,
        epoch_length=args.epoch_length,
        num_workers=args.num_workers,
        save_checkpoint_frequency=args.save_checkpoint_frequency,
        eval_period_iterations=args.eval_period_iterations,
        learning_rates=args.learning_rates,
        weight_decays=args.weight_decays,
        autocast_dtype=autocast_dtype,
        resume=not args.no_resume,
        classifier_fpath=args.classifier_fpath,
        val_metric_type=args.val_metric_type,
        test_metric_types=args.test_metric_types,
        loss_type=args.loss_type,
        bag_of_channels=args.bag_of_channels,
        leave_one_out_dataset=args.leave_one_out_dataset,
        crop_size=args.crop_size,
        resize_size=args.resize_size,
        n_last_blocks=args.n_last_blocks,
        avgpool=args.avgpool,
        scheduler=args.scheduler,
    )
    return 0


if __name__ == "__main__":
    description = "DINOv2 linear_cell_dino evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))
