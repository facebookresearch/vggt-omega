# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC licence,
# found in the LICENSE_CELL_DINO_CODE file in the root directory of this source tree.

import logging
from typing import Callable, Dict, Optional, Any, List

import torch
from torch import nn
from torchmetrics import MetricCollection

from dinov2.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader
from dinov2.data import NoOpAccumulator, ResultsAccumulator
import dinov2.distributed as distributed
from dinov2.logging import MetricLogger
from enum import Enum
from torch.utils.data import Subset
from torchvision.datasets.vision import StandardTransform
import numpy as np
from torch.nn.functional import one_hot, softmax

logger = logging.getLogger("dinov2")


class LossType(Enum):
    CROSS_ENTROPY = "cross_entropy"
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"


class BagOfChannelsModelWithNormalize(nn.Module):
    def __init__(self, model, autocast_ctx, avgpool, n_last_blocks=1):
        super().__init__()
        self.model = model
        self.autocast_ctx = autocast_ctx
        self.n_last_blocks = n_last_blocks
        self.avgpool = avgpool

    def forward(self, samples):
        with self.autocast_ctx():
            features = self.model.get_intermediate_layers(samples, self.n_last_blocks, return_class_token=True)
            output = create_linear_input(features, self.avgpool, use_n_blocks=self.n_last_blocks)
            return nn.functional.normalize(output, dim=1, p=2)


@torch.inference_mode()
def evaluate_with_accumulate(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    test_mode: bool = False,
    accumulate_results: bool = False,
    leave_one_out: bool = False,
):
    model.eval()

    if test_mode:
        output_tensor = {k: [] for k in postprocessors.keys()}
        target_tensor = {k: [] for k in postprocessors.keys()}

    if criterion is not None:
        criterion.eval()

    accumulator_class = ResultsAccumulator if accumulate_results else NoOpAccumulator
    accumulators = {k: accumulator_class() for k in postprocessors.keys()}

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for samples, targets, *_ in metric_logger.log_every(data_loader, 10, header):
        if isinstance(targets, list):
            index = targets[0]
            targets = targets[1]
            samples, targets, index = samples[index >= 0], targets[index >= 0], index[index >= 0]
            if len(index) == 0:
                continue

        outputs = samples.to(device) if leave_one_out else model(samples.to(device))
        targets = targets.to(device)

        if criterion is not None:
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())

        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets)
            metric.update(**metric_inputs)
            if test_mode:
                output_tensor[k].append(metric_inputs["preds"])
                target_tensor[k].append(metric_inputs["target"])
            accumulators[k].update(preds=metric_inputs["preds"], target=metric_inputs["target"], index=index)

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # accumulator.accumulate() returns None for the NoOpAccumulator
    accumulated_results = {k: accumulator.accumulate() for k, accumulator in accumulators.items()}
    if test_mode:
        for k in postprocessors.keys():
            output_tensor[k] = torch.cat(output_tensor[k])
            target_tensor[k] = torch.cat(target_tensor[k])
        accumulated_results = {k: [output_tensor[k], target_tensor[k]] for k in postprocessors.keys()}

    if accumulate_results:
        return metric_logger_stats, stats
    return metric_logger_stats, stats, accumulated_results


def all_gather_and_flatten(tensor_rank):
    tensor_all_ranks = torch.empty(
        distributed.get_global_size(),
        *tensor_rank.shape,
        dtype=tensor_rank.dtype,
        device=tensor_rank.device,
    )
    tensor_list = list(tensor_all_ranks.unbind(0))
    torch.distributed.all_gather(tensor_list, tensor_rank.contiguous())
    return tensor_all_ranks.flatten(end_dim=1)


def extract_features_cell_dino(
    model, dataset, batch_size, num_workers, gather_on_cpu=False, shuffle=False, avgpool=False
):
    dataset_with_enumerated_targets = DatasetWithEnumeratedTargets(dataset)
    sample_count = len(dataset_with_enumerated_targets)
    data_loader = make_data_loader(
        dataset=dataset_with_enumerated_targets,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=shuffle,
    )
    return extract_features_with_dataloader_cell_dino(model, data_loader, sample_count, gather_on_cpu, avgpool=avgpool)


@torch.inference_mode()
def extract_features_with_dataloader_cell_dino(model, data_loader, sample_count, gather_on_cpu=False, avgpool=False):
    gather_device = torch.device("cpu") if gather_on_cpu else torch.device("cuda")
    metric_logger = MetricLogger(delimiter="  ")
    features, all_labels = None, None
    for samples, (index, labels_rank) in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        labels_rank = labels_rank.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        feat = model(samples)
        if isinstance(samples, list) or isinstance(feat, tuple):
            features_rank = create_linear_input(feat, avgpool=avgpool)
        else:
            features_rank = feat

        # init storage feature matrix
        if features is None:
            features = torch.zeros(sample_count, features_rank.shape[-1], device=gather_device)
            labels_shape = list(labels_rank.shape)
            labels_shape[0] = sample_count
            all_labels = torch.full(labels_shape, fill_value=-1, device=gather_device)
            logger.info(f"Storing features into tensor of shape {features.shape}")

        # share indexes, features and labels between processes
        index_all = all_gather_and_flatten(index).to(gather_device)
        features_all_ranks = all_gather_and_flatten(features_rank).to(gather_device)
        labels_all_ranks = all_gather_and_flatten(labels_rank).to(gather_device)

        # update storage feature matrix
        if len(index_all) > 0:
            features.index_copy_(0, index_all, features_all_ranks)
            all_labels.index_copy_(0, index_all, labels_all_ranks)

    logger.info(f"Features shape: {tuple(features.shape)}")
    logger.info(f"Labels shape: {tuple(all_labels.shape)}")

    assert torch.all(all_labels > -1)

    return features, all_labels


def create_linear_input(x_tokens_list, avgpool=False, use_n_blocks=1):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat(
        [class_token for _, class_token in intermediate_output], dim=-1
    )  # concatenate class tokens of the last n blocks
    if avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=-2).reshape(
                    intermediate_output[-1][0].shape[0], -1
                ),  # average pooling of patch tokens: average over N, then concatenate channels if single-channel patch model
            ),
            dim=-1,
        )  # concatenate average pooling of patch tokens to concatenated patch tokens
    output = output.reshape(output.shape[0], -1)

    return output.float()


def get_target_transform(dataset) -> Optional[Callable]:
    if hasattr(dataset, "transforms"):
        if isinstance(dataset.transforms, StandardTransform):
            return dataset.transforms.target_transform
        raise ValueError("Dataset has a non-standard .transforms property")
    if hasattr(dataset, "target_transform"):
        return dataset.target_transform
    return None


def get_labels(dataset) -> torch.Tensor:
    """
    Get the labels of a classification dataset, as a Tensor, using the `get_targets` method
    if it is present or loading the labels one by one with `get_target`, if it exists.
    If the dataset has a target transform, iterate over the whole dataset to get the
    transformed labels for each element, then stack them as a torch tensor.
    """
    logger.info("Getting dataset labels ...")
    if hasattr(dataset, "get_targets") or hasattr(dataset, "get_target"):
        if hasattr(dataset, "get_targets"):  # Returns a np.array
            labels = dataset.get_targets()
        elif hasattr(dataset, "get_target"):
            labels = [dataset.get_target(i) for i in range(len(dataset))]
        target_transform = get_target_transform(dataset)
        if target_transform is not None:
            labels = [target_transform(label) for label in labels]
    else:
        # Target transform is applied in this case
        labels = [dataset[i][1] for i in range(len(dataset))]
    return torch.stack([torch.tensor(label, dtype=int) for label in labels])


def get_num_classes(dataset) -> int:
    """
    Get the labels of a dataset and compute the number of classes
    """
    labels = get_labels(dataset)
    if len(labels.shape) > 1:
        return int(labels.shape[1])
    return int(labels.max() + 1)


def average_metrics(eval_metrics_dict: dict[Any, dict[str, torch.Tensor]], ignore_keys: List[str] = []):
    """
    Function that computes the average and the std on a metrics dict.
    A linear evaluation dictionary contains "best_classifier",
    so this specific key is removed for computing aggregated metrics.
    """
    output_metrics_dict = {}
    metrics = [metric for metric in eval_metrics_dict[0].keys() if metric not in ignore_keys]
    for metric in metrics:
        stats_tensor = torch.tensor([stat[metric] for stat in eval_metrics_dict.values()])
        output_metrics_dict[metric + "_mean"] = stats_tensor.mean().item()
        output_metrics_dict[metric + "_std"] = torch.std(stats_tensor).item()

    return output_metrics_dict


def create_class_indices_mapping(labels: torch.Tensor) -> dict[int, torch.Tensor]:
    """
    Efficiently creates a mapping between the labels and tensors containing
    the indices of all the dataset elements that share this label.
    In the case of multiple labels, it is not guaranteed that there
    will be exactly the specified percentage of labels.
    """
    if len(labels.shape) > 1:  # labels are a one-hot encoding
        assert len(labels.shape) == 2
        sorted_labels, indices = torch.nonzero(labels.T, as_tuple=True)
    else:
        sorted_labels, indices = torch.sort(labels, stable=True)
    unique_labels, counts = torch.unique_consecutive(sorted_labels, return_counts=True)
    mapping = dict(zip(unique_labels.tolist(), torch.split(indices, counts.tolist())))
    return mapping


def _shuffle_dataset(dataset: torch.Tensor, seed: int = 0):
    """
    Shuffling a dataset by subsetting it with a random permutation of its indices
    """
    random_generator = torch.Generator()
    random_generator.manual_seed(seed)
    random_indices = torch.randperm(len(dataset), generator=random_generator)
    return Subset(dataset, random_indices)


def _subset_dataset_per_class(
    class_indices_mapping: dict[int, torch.Tensor],
    n_or_percent_per_class: float,
    dataset_size: int,
    seed: int = 0,
    is_percent: bool = False,
) -> torch.Tensor:
    """
    Helper function to select a percentage of a dataset, equally distributed across classes,
    or to take the same number of elements from each class of the dataset.
    Returns a boolean mask tensor being True at indices of selected elements
    """

    random_generator = torch.Generator()
    random_generator.manual_seed(seed)

    final_indices_bool = torch.zeros(dataset_size, dtype=bool)
    for class_indices in class_indices_mapping.values():
        # Select at least one element
        n_for_class = max(int(len(class_indices) * n_or_percent_per_class), 1) if is_percent else n_or_percent_per_class
        assert isinstance(n_for_class, int)
        filtered_index = torch.randperm(len(class_indices), generator=random_generator)[:n_for_class]
        final_indices_bool[class_indices[filtered_index]] = True
    return final_indices_bool


def _multilabel_rebalance_subset(
    class_indices_mapping: dict[int, torch.Tensor],
    n_or_percent_per_class: float,
    labels: torch.Tensor,
    indices_bool: torch.Tensor,
    dataset_size: int,
    seed: int = 0,
) -> torch.Tensor:
    """
    Helper function to refine a subset of a multi-label dataset (indices_bool)
    to better match a target percentage of labels.
    Returns a boolean mask tensor being True at indices of selected elements.
    """

    # Compute the number of selected labels in indices_bool
    num_total_labels = labels.sum()
    num_wanted_labels = int(num_total_labels * n_or_percent_per_class)
    num_selected_labels = (labels[indices_bool] > 0).sum()
    logger.info(f" {num_selected_labels} labels instead of {num_wanted_labels}")

    # Compute a new percentage and new set selecting less images, therefore less labels, to match approximatelly the exact percentage of labels selected
    n_or_percent_per_class = n_or_percent_per_class / (num_selected_labels / num_wanted_labels)
    final_indices_bool = _subset_dataset_per_class(
        class_indices_mapping, n_or_percent_per_class, dataset_size, seed, True
    )

    # Compute the number of labels finally used
    num_selected_labels = (labels[final_indices_bool] > 0).sum()
    logger.info(f" {num_selected_labels} labels instead of {num_wanted_labels}")

    return final_indices_bool


def split_train_val_datasets(train_dataset, split_percentage: float = 0.1, shuffle_train: bool = True):
    """
    Splitting a percent of the train dataset to choose hyperparameters, taking the same percentage for each class.
    If `shuffle` is False, taking the first elements of each class as the validaton set.
    """
    assert 0 < split_percentage < 1
    logger.info(f"Selecting {int(split_percentage * 100)}% of the train dataset as the validation set")
    if shuffle_train:
        logger.info("Shuffling train dataset before splitting in train and validation sets")
        train_dataset = _shuffle_dataset(train_dataset)
    train_labels = get_labels(train_dataset)
    class_indices_mapping = create_class_indices_mapping(train_labels)
    val_mask = torch.zeros(len(train_labels), dtype=bool)
    for class_indices in class_indices_mapping.values():
        # If there is only one element, it goes in the train set
        n_for_val = max(1, int(split_percentage * len(class_indices))) if len(class_indices) > 1 else 0
        val_mask[class_indices[:n_for_val]] = True

    val_dataset = Subset(train_dataset, val_mask.nonzero().flatten())
    train_dataset = Subset(train_dataset, (~val_mask).nonzero().flatten())
    return train_dataset, val_dataset


def create_train_dataset_dict(
    train_dataset,
    few_shot_eval: bool = False,
    few_shot_k_or_percent=None,
    few_shot_n_tries: int = 1,
) -> dict[int, dict[int, Any]]:
    """
    Randomly split a dataset for few-shot evaluation, with `few_shot_k_or_percent` being
    n elements or x% of a class. Produces a dict, which keys are number of random "tries"
    and values are the dataset subset for this "try".

    Format is {"nth-try": dataset}
    """
    if few_shot_eval is False:
        assert few_shot_k_or_percent is None
        assert few_shot_n_tries == 1
        return {0: train_dataset}

    assert few_shot_k_or_percent is not None
    train_labels = get_labels(train_dataset)
    class_indices_mapping = create_class_indices_mapping(train_labels)
    train_dataset_dict: dict[int, Any] = {}
    is_percent = few_shot_k_or_percent < 1
    if not is_percent:
        few_shot_k_or_percent = int(few_shot_k_or_percent)

    for t in range(few_shot_n_tries):
        t_subset_bool = _subset_dataset_per_class(
            class_indices_mapping=class_indices_mapping,
            n_or_percent_per_class=few_shot_k_or_percent,
            dataset_size=len(train_labels),
            is_percent=is_percent,
            seed=t,
        )
        if len(train_labels.shape) > 1 and is_percent:
            t_subset_bool = _multilabel_rebalance_subset(
                class_indices_mapping=class_indices_mapping,
                n_or_percent_per_class=few_shot_k_or_percent,
                dataset_size=len(train_labels),
                labels=train_labels,
                indices_bool=t_subset_bool,
                seed=t,
            )
        train_dataset_dict[t] = Subset(train_dataset, t_subset_bool.nonzero().flatten())
    return train_dataset_dict


def extract_features_for_dataset_dict(
    model,
    dataset_dict: dict[int, dict[int, Any]],
    batch_size: int,
    num_workers: int,
    gather_on_cpu=False,
    avgpool=False,
) -> dict[int, dict[str, torch.Tensor]]:
    """
    Extract features for each subset of dataset in the context of few-shot evaluations
    """
    few_shot_data_dict: dict[int, dict[str, torch.Tensor]] = {}
    for try_n, dataset in dataset_dict.items():
        features, labels = extract_features_cell_dino(
            model, dataset, batch_size, num_workers, gather_on_cpu=gather_on_cpu, avgpool=avgpool
        )
        few_shot_data_dict[try_n] = {"train_features": features, "train_labels": labels}
    return few_shot_data_dict


def pad_multilabel_and_collate(batch, pad_value=-1):
    """
    This method pads and collates a batch of (image, (index, target)) tuples, coming from
    DatasetWithEnumeratedTargets, with targets that are list of potentially varying sizes.
    The targets are padded to the length of the longest target list in the batch.
    """
    maxlen = max(len(targets) for _, (_, targets) in batch)
    padded_batch = [
        (image, (index, np.pad(targets, (0, maxlen - len(targets)), constant_values=pad_value)))
        for image, (index, targets) in batch
    ]
    return torch.utils.data.default_collate(padded_batch)


class KnnModule(torch.nn.Module):
    """
    Gets knn of test features from all processes on a chunk of the train features

    Each rank gets a chunk of the train features as well as a chunk of the test features.
    In `compute_neighbors`, for each rank one after the other, its chunk of test features
    is sent to all devices, partial knns are computed with each chunk of train features
    then collated back on the original device.
    """

    def __init__(self, train_features, train_labels, nb_knn, T, device, num_classes=1000):
        super().__init__()

        self.global_rank = distributed.get_global_rank()
        self.global_size = distributed.get_global_size()

        self.device = device
        self.train_features_rank_T = train_features.chunk(self.global_size)[self.global_rank].T.to(self.device)
        # Labels can either be integers, or in a one-hot format
        self.candidates = train_labels.chunk(self.global_size)[self.global_rank].unsqueeze(0).to(self.device)
        self.nb_knn = nb_knn
        self.max_k = max(self.nb_knn)
        self.T = T
        self.num_classes = num_classes

    def _get_knn_sims_and_labels(self, similarity, train_labels):
        topk_sims, indices = similarity.topk(self.max_k, largest=True, sorted=True)
        if len(train_labels.shape) == 3:  # If the labels are in one_hot format
            indices = indices.unsqueeze(2).expand(-1, -1, self.num_classes)  # Orignally [bs, max_k]
        neighbors_labels = torch.gather(train_labels, 1, indices)
        return topk_sims, neighbors_labels

    def _similarity_for_rank(self, features_rank, source_rank):
        # Send the features from `source_rank` to all ranks
        broadcast_shape = torch.tensor(features_rank.shape).to(self.device)
        torch.distributed.broadcast(broadcast_shape, source_rank)

        broadcasted = features_rank
        if self.global_rank != source_rank:
            broadcasted = torch.zeros(*broadcast_shape, dtype=features_rank.dtype, device=self.device)
        torch.distributed.broadcast(broadcasted, source_rank)

        # Compute the neighbors for `source_rank` among `train_features_rank_T`
        similarity_rank = torch.mm(broadcasted, self.train_features_rank_T)
        candidate_labels = self.candidates.expand(len(similarity_rank), *self.candidates.shape[1:])
        return self._get_knn_sims_and_labels(similarity_rank, candidate_labels)

    def _gather_all_knn_for_rank(self, topk_sims, neighbors_labels, target_rank):
        # Gather all neighbors for `target_rank`
        topk_sims_rank = retrieved_rank = None
        if self.global_rank == target_rank:
            topk_sims_rank = [torch.zeros_like(topk_sims) for _ in range(self.global_size)]
            retrieved_rank = [torch.zeros_like(neighbors_labels) for _ in range(self.global_size)]

        torch.distributed.gather(topk_sims, topk_sims_rank, dst=target_rank)
        torch.distributed.gather(neighbors_labels, retrieved_rank, dst=target_rank)

        if self.global_rank == target_rank:
            # Perform a second top-k on the k * global_size retrieved neighbors
            topk_sims_rank = torch.cat(topk_sims_rank, dim=1)
            retrieved_rank = torch.cat(retrieved_rank, dim=1)
            results = self._get_knn_sims_and_labels(topk_sims_rank, retrieved_rank)
            return results
        return None

    def compute_neighbors(self, features_rank):
        for rank in range(self.global_size):
            topk_sims, neighbors_labels = self._similarity_for_rank(features_rank, rank)
            results = self._gather_all_knn_for_rank(topk_sims, neighbors_labels, rank)
            if results is not None:
                topk_sims_rank, neighbors_labels_rank = results
        return topk_sims_rank, neighbors_labels_rank

    def forward(self, features_rank):
        """
        Compute the results on all values of `self.nb_knn` neighbors from the full `self.max_k`
        """
        assert all(k <= self.max_k for k in self.nb_knn)

        topk_sims, neighbors_labels = self.compute_neighbors(features_rank)
        batch_size = neighbors_labels.shape[0]
        topk_sims_transform = softmax(topk_sims / self.T, 1)
        voting_coefficient = topk_sims_transform.view(batch_size, -1, 1)
        if len(neighbors_labels.shape) == 2:  # If the labels are not yet one hot
            neighbors_labels = one_hot(neighbors_labels, num_classes=self.num_classes)
        matmul = torch.mul(neighbors_labels, voting_coefficient)
        probas_for_k = {k: torch.sum(matmul[:, :k, :], 1) for k in self.nb_knn}
        return probas_for_k
