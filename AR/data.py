from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from AR.config import PipelineConfig
from AR.schema import DatasetSchema


@dataclass
class ArtifactBundle:
    schema: DatasetSchema
    train_tokens: np.ndarray
    val_tokens: np.ndarray
    test_tokens: np.ndarray


class TokenRowDataset(Dataset[torch.Tensor]):
    def __init__(self, token_array: np.ndarray) -> None:
        self.token_array = token_array

    def __len__(self) -> int:
        return int(self.token_array.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.token_array[index]
        return torch.from_numpy(np.asarray(row, dtype=np.int64))


def load_schema(path: str | Path) -> DatasetSchema:
    return DatasetSchema.load(path)


def load_artifacts(config: PipelineConfig, mmap_train: bool = True) -> ArtifactBundle:
    schema = load_schema(config.paths.schema_path)
    train_mode = "r" if mmap_train else None
    train_tokens = np.load(config.paths.train_tokens_path, mmap_mode=train_mode)
    val_tokens = np.load(config.paths.val_tokens_path)
    test_tokens = np.load(config.paths.test_tokens_path)
    return ArtifactBundle(
        schema=schema,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        test_tokens=test_tokens,
    )
