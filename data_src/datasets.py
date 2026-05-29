"""Dataset registry and construction utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from configs.configuration import DatasetConfig

from . import utils


LOGGER = logging.getLogger(__name__)


@dataclass
class DatasetFactoryOutput:
    dataset: Dataset
    resolution: int
    in_channels: int
    postprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


DatasetFactory = Callable[[DatasetConfig], DatasetFactoryOutput]

@dataclass
class DatasetBundle:
    name: str
    dataset: Dataset
    dataloader: DataLoader
    resolution: int
    in_channels: int
    split: str
    postprocess: Callable[[torch.Tensor], torch.Tensor]


DATASET_REGISTRY: Dict[str, DatasetFactory] = {}


def register_dataset(
    name: str,
) -> Callable[[DatasetFactory], DatasetFactory]:
    """Register a dataset builder."""

    def decorator(factory: DatasetFactory) -> DatasetFactory:
        DATASET_REGISTRY[name.lower()] = factory
        return factory

    return decorator



def _build_indices_per_class(torch_dataset, num_classes: int = 1000):
    labels = getattr(torch_dataset, "targets", None)
    if labels is None:
        labels = getattr(torch_dataset, "labels", None)
    if labels is None:
        raise ValueError("dataset has no 'targets' or 'labels' attribute")

    labels = torch.as_tensor(labels, dtype=torch.long)
    num_classes_auto = int(labels.max().item() + 1)
    if num_classes_auto < num_classes:
        num_classes = num_classes_auto

    indices_per_class = [[] for _ in range(num_classes)]
    for idx, y in enumerate(labels.tolist()):
        if 0 <= y < num_classes:
            indices_per_class[y].append(idx)
    return indices_per_class



def build_dataset(cfg: DatasetConfig) -> DatasetBundle:
    factory = DATASET_REGISTRY.get(cfg.name.lower())
    if factory is None:
        raise ValueError(f"Unsupported dataset: {cfg.name}")

    resources = factory(cfg)
    dataset = resources.dataset
    if not isinstance(dataset, Dataset):
        raise TypeError(
            f"Dataset factory for '{cfg.name}' must return a torch.utils.data.Dataset instance"
        )

    dataset = utils.maybe_apply_subset(dataset, cfg.subset_size)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    postprocess = resources.postprocess or utils.identity

    return DatasetBundle(
        name=cfg.name,
        dataset=dataset,
        dataloader=dataloader,
        resolution=resources.resolution,
        in_channels=resources.in_channels,
        split=cfg.split,
        postprocess=postprocess,
    )

