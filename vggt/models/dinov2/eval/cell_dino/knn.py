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
from typing import List, Optional, Any
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import pandas as pd
from sklearn.metrics import f1_score

import dinov2.distributed as distributed
from dinov2.data import make_dataset, DatasetWithEnumeratedTargets, SamplerType, make_data_loader
from dinov2.data.cell_dino.transforms import NormalizationType, make_classification_eval_cell_transform
from dinov2.eval.metrics import build_metric, MetricType
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model

from dinov2.data import ResultsAccumulator
from dinov2.eval.utils import ModelWithNormalize
from dinov2.eval.cell_dino.utils import (
    BagOfChannelsModelWithNormalize,
    extract_features_cell_dino,
    average_metrics,
    create_train_dataset_dict,
    get_num_classes,
    extract_features_for_dataset_dict,
    evaluate_with_accumulate,
    KnnModule,
)
from dinov2.eval.knn import DictKeysModule
from torch.utils.data import Subset as SubsetEx
from torch.utils.data import ConcatDataset as ConcatDatasetEx


logger = logging.getLogger("dinov2")


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
        "--nb_knn",
        nargs="+",
        type=int,
        help="Number of NN to use. 20 is usually working the best.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature used in the voting coefficient",
    )
    parser.add_argument(
        "--gather-on-cpu",
        action="store_true",
        help="Whether to gather the train features on cpu, slower"
        "but useful to avoid OOM for large datasets (e.g. ImageNet22k).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--n-per-class-list",
        nargs="+",
        type=int,
        help="Number to take per class",
    )
    parser.add_argument(
        "--n-tries",
        type=int,
        help="Number of tries",
    )
    parser.add_argument(
        "--leave-one-out-dataset",
        type=str,
        help="Path with indexes to use the leave one out strategy for CHAMMI_CP task 3 and CHAMMI_HPA task 4",
    )
    parser.add_argument(
        "--bag-of-channels",
        action="store_true",
        help='Whether to use the "bag of channels" channel adaptive strategy',
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
        "--metric-type",
        type=MetricType,
        choices=list(MetricType),
        help="Validation metric",
    )
    parser.add_argument(
        "--avgpool",
        action="store_true",
        help="Whether to use average pooling of path tokens in addition to CLS tokens",
    )

    parser.set_defaults(
        train_dataset_str="ImageNet:split=TRAIN",
        val_dataset_str="ImageNet:split=VAL",
        nb_knn=[1],
        temperature=0.07,
        batch_size=256,
        resize_size=0,
    )
    return parser


class SequentialWithKwargs(torch.nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, input, **kwargs):

        input = self[0](input, **kwargs)
        for module in self[1:]:
            input = module(input)
        return input


def create_train_test_dataset_dict_leave_one_out(
    train_dataset,
    test_dataset,
) -> dict[int, dict[int, Any]]:
    """
    This function implements a train dataset dictionary with the leave-one-out (LOO) method.
    Specifically, given a train dataset and test dataset, it creates a train dataset for each
    test dataset point, which is a combination of train+test dataset except for this specific data point.
    At the end, it contains len(test_dataset) key and value pairs.

    Format is {"nth-test-sample": dataset_without_test_sample}
    """
    train_dataset_dict: dict[int, Any] = {}
    test_size = len(test_dataset)

    for test_sample_index in range(test_size):
        test_indices_bool = torch.ones(test_size, dtype=bool)
        test_indices_bool[test_sample_index] = False
        train_dataset_dict[test_sample_index] = ConcatDatasetEx(
            [train_dataset, SubsetEx(test_dataset, test_indices_bool.nonzero().flatten())]
        )

    return train_dataset_dict


def eval_knn_with_leave_one_out(
    model, leave_one_out_dataset, train_dataset, test_dataset, metric_type, nb_knn, temperature, batch_size, num_workers
):
    num_classes = get_num_classes(test_dataset)
    train_dataset_dict = create_train_dataset_dict(train_dataset)
    test_dataset_dict = create_train_dataset_dict(test_dataset)

    logger.info("Extracting features for train set...")
    train_data_dict = extract_features_for_dataset_dict(
        model, train_dataset_dict, batch_size, num_workers, gather_on_cpu=True
    )
    test_data_dict = extract_features_for_dataset_dict(
        model, test_dataset_dict, batch_size, num_workers, gather_on_cpu=True
    )

    train_features = train_data_dict[0]["train_features"]
    train_labels = train_data_dict[0]["train_labels"]
    test_features = test_data_dict[0]["train_features"]
    test_labels = test_data_dict[0]["train_labels"]

    metric_collection = build_metric(metric_type, num_classes=3)

    device = torch.cuda.current_device()
    partial_knn_module = partial(KnnModule, T=temperature, device=device, num_classes=num_classes)

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
        leave_one_out_indices[leave_one_out_label] = torch.tensor(
            metadata[metadata[leave_one_out_label_type] == leave_one_out_label].index.values
        )

    # ============ evaluation ... ============
    logger.info("Start the k-NN classification.")

    eval_metrics_dict = {}
    postprocessors, metrics = {k: DictKeysModule([k]) for k in nb_knn}, {
        k: metric_collection.clone().to(device) for k in nb_knn
    }
    for metric_key in metrics.keys():
        metrics[metric_key] = metrics[metric_key].to(device)

    accumulator_class = ResultsAccumulator
    accumulators = {k: accumulator_class() for k in postprocessors.keys()}
    all_preds = []
    all_target = []

    for loo_label, loo_indices in leave_one_out_indices.items():
        logger.info(f"Evaluating on test sample {loo_label}")
        loo_for_training_indices = torch.ones(test_features.shape[0], dtype=bool)
        loo_for_training_indices[loo_indices] = False
        train_features_sample = torch.cat([train_features, test_features[loo_for_training_indices]])
        train_labels_sample = torch.cat([train_labels, test_labels[loo_for_training_indices]])
        logger.info(f"Train shape {train_features_sample.shape}, Test shape {test_features[loo_indices].shape}")
        logger.info(
            f"Train values {train_labels_sample.unique(return_counts=True)}, Test shape {test_labels[loo_indices].unique(return_counts=True)}"
        )
        knn_module = partial_knn_module(
            train_features=train_features_sample, train_labels=train_labels_sample, nb_knn=nb_knn
        )

        output = knn_module(test_features[loo_indices].to(device))
        all_preds.append(output[1])
        all_target.append(test_labels[loo_indices])
        output[1] = output[1][:, 4:]
        transformed_test_labels = test_labels[loo_indices] - 4
        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](output, transformed_test_labels.to(device))
            metric.update(**metric_inputs)
            accumulators[k].update(
                preds=metric_inputs["preds"], target=metric_inputs["target"], index=loo_indices.to(device)
            )

    all_preds = torch.cat(all_preds).cpu().detach().numpy()

    all_preds = np.argmax(all_preds, axis=1)
    all_target = torch.cat(all_target).cpu().detach().numpy()

    f1 = f1_score(all_target, all_preds, average="macro", labels=[4, 5, 6])
    logger.info(f"Real f1 score: {f1}")
    eval_metrics = {
        k: metric.compute() for k, metric in metrics.items()
    }  # next erased by the real f1 score computed above

    for k in nb_knn:
        if k not in eval_metrics_dict:
            eval_metrics_dict[k] = {}
        eval_metrics_dict[k] = {metric: f1 * 100.0 for metric, v in eval_metrics[k].items()}

    if len(train_data_dict) > 1:
        return {k: average_metrics(eval_metrics_dict[k]) for k in eval_metrics_dict.keys()}

    return {k: eval_metrics_dict[k] for k in eval_metrics_dict.keys()}


def eval_knn_with_model(
    model,
    output_dir,
    train_dataset_str,
    val_dataset_str,
    nb_knn=(10, 20, 100, 200),
    temperature=0.07,
    autocast_dtype=torch.float,
    metric_type=MetricType.MEAN_ACCURACY,
    transform=None,
    resize_size=256,
    crop_size=224,
    batch_size=256,
    num_workers=5,
    leave_one_out_dataset="",
    bag_of_channels=False,
    avgpool=False,
):
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    if bag_of_channels:
        model = BagOfChannelsModelWithNormalize(model, autocast_ctx, avgpool)
    else:
        model = ModelWithNormalize(model)
    if leave_one_out_dataset == "" or leave_one_out_dataset is None:
        leave_one_out = False
    else:
        leave_one_out = True

    cudnn.benchmark = True
    transform = make_classification_eval_cell_transform(
        normalization_type=NormalizationType.SELF_NORM_CENTER_CROP, resize_size=resize_size, crop_size=crop_size
    )

    train_dataset = make_dataset(dataset_str=train_dataset_str, transform=transform)
    results_dict = {}
    test_dataset = make_dataset(dataset_str=val_dataset_str, transform=transform)

    with torch.cuda.amp.autocast(dtype=autocast_dtype):
        if leave_one_out:
            results_dict_knn = eval_knn_with_leave_one_out(
                model=model,
                leave_one_out_dataset=leave_one_out_dataset,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                metric_type=metric_type,
                nb_knn=nb_knn,
                temperature=temperature,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        else:
            results_dict_knn = eval_knn(
                model=model,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                metric_type=metric_type,
                nb_knn=nb_knn,
                temperature=temperature,
                batch_size=batch_size,
                num_workers=num_workers,
            )

    for knn_ in results_dict_knn.keys():
        top1 = results_dict_knn[knn_]["top-1"]
        results_dict[f"{val_dataset_str}_{knn_} Top 1"] = top1
        results_string = f"{val_dataset_str} {knn_} NN classifier result: Top1: {top1:.2f}"
        if "top-5" in results_dict_knn[knn_]:
            top5 = results_dict_knn[knn_]["top-5"]
            results_dict[f"{val_dataset_str}_{knn_} Top 5"] = top5
            results_string += f"Top5: {top5:.2f}"
        logger.info(results_string)

    metrics_file_path = os.path.join(output_dir, "results_eval_knn.json")
    with open(metrics_file_path, "a") as f:
        for k, v in results_dict.items():
            f.write(json.dumps({k: v}) + "\n")

    if distributed.is_enabled():
        torch.distributed.barrier()
    return results_dict


def eval_knn(
    model,
    train_dataset,
    test_dataset,
    metric_type,
    nb_knn,
    temperature,
    batch_size,
    num_workers,
    few_shot_eval=False,
    few_shot_k_or_percent=None,
    few_shot_n_tries=1,
):
    num_classes = get_num_classes(train_dataset)
    train_dataset_dict = create_train_dataset_dict(
        train_dataset,
        few_shot_eval=few_shot_eval,
        few_shot_k_or_percent=few_shot_k_or_percent,
        few_shot_n_tries=few_shot_n_tries,
    )

    logger.info("Extracting features for train set...")

    train_data_dict: dict[int, dict[str, torch.Tensor]] = {}
    for try_n, dataset in train_dataset_dict.items():
        features, labels = extract_features_cell_dino(model, dataset, batch_size, num_workers, gather_on_cpu=True)
        train_data_dict[try_n] = {"train_features": features, "train_labels": labels}

    test_data_loader = make_data_loader(
        dataset=DatasetWithEnumeratedTargets(
            test_dataset, pad_dataset=True, num_replicas=distributed.get_global_size()
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=True,
        collate_fn=None,
    )
    metric_collection = build_metric(metric_type, num_classes=num_classes)

    device = torch.cuda.current_device()
    partial_knn_module = partial(
        KnnModule,
        T=temperature,
        device=device,
        num_classes=num_classes,
    )

    # ============ evaluation ... ============
    logger.info("Start the k-NN classification.")
    eval_metrics_dict = {}

    for try_ in train_data_dict.keys():
        train_features, train_labels = train_data_dict[try_]["train_features"], train_data_dict[try_]["train_labels"]
        k_list = sorted(set([el if el < len(train_features) else len(train_features) for el in nb_knn]))
        knn_module = partial_knn_module(train_features=train_features, train_labels=train_labels, nb_knn=k_list)
        postprocessors, metrics = {k: DictKeysModule([k]) for k in k_list}, {
            k: metric_collection.clone() for k in k_list
        }
        _, eval_metrics, _ = evaluate_with_accumulate(
            SequentialWithKwargs(model, knn_module),
            test_data_loader,
            postprocessors,
            metrics,
            device,
            accumulate_results=False,
        )
        for k in k_list:
            if k not in eval_metrics_dict:
                eval_metrics_dict[k] = {}
            eval_metrics_dict[k][try_] = {metric: v.item() * 100.0 for metric, v in eval_metrics[k].items()}

    if len(train_data_dict) > 1:
        return {k: average_metrics(eval_metrics_dict[k]) for k in eval_metrics_dict.keys()}

    return {k: eval_metrics_dict[k][0] for k in eval_metrics_dict.keys()}


def main(args):
    model, autocast_dtype = setup_and_build_model(args)
    eval_knn_with_model(
        model=model,
        output_dir=args.output_dir,
        train_dataset_str=args.train_dataset_str,
        val_dataset_str=args.val_dataset_str,
        nb_knn=args.nb_knn,
        temperature=args.temperature,
        autocast_dtype=autocast_dtype,
        transform=None,
        metric_type=args.metric_type,
        batch_size=args.batch_size,
        num_workers=5,
        leave_one_out_dataset=args.leave_one_out_dataset,
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        avgpool=args.avgpool,
        bag_of_channels=args.bag_of_channels,
    )
    return 0


if __name__ == "__main__":
    description = "k-NN evaluation on models trained with bag of channel strategy or cell dino"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))
