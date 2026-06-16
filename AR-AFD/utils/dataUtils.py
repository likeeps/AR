#!/usr/bin/env python
# coding: utf-8

"""Shared tabular preprocessing and dataset metadata utilities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

# -----------------------------------------------------------------------------
# Dataset schema
# -----------------------------------------------------------------------------

# Continuous numeric columns (high cardinality, treat as continuous)
ADULT_CONTINUOUS_COLS = [
    "fnlwgt",  # Population weight, very high cardinality (ratio ~0.93)
]

# Discrete numeric columns (medium cardinality, preserve numeric semantics)
ADULT_DISCRETE_NUMERIC_COLS = [
    "age",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "education_num",  # Ordinal, corresponds to education
]

# True categorical columns (low cardinality)
ADULT_CATEGORICAL_COLS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

# Backward-compatible alias used by preprocessing code
ADULT_NUMERIC_COLS = ADULT_CONTINUOUS_COLS + ADULT_DISCRETE_NUMERIC_COLS

ADULT_TARGET_COL = "outcome"

# Keep original CSV column order for compatibility
ADULT_ALL_COLS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "outcome"
]

def resolve_project_path(project_path: str | None = None) -> str:
    if project_path:
        return project_path
    # Always return the project root directory (parent of utils directory)
    current_file = os.path.abspath(__file__)
    utils_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(utils_dir)
    return project_root


def infer_project_path(project_path: str | None = None, required_child: str = "traindata") -> str:
    candidates = []
    try:
        candidates.append(resolve_project_path(project_path))
    except Exception:
        pass
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.extend([os.path.dirname(current_dir), current_dir])
    for candidate in candidates:
        if candidate and os.path.isdir(os.path.join(candidate, required_child)):
            return candidate
    return candidates[0] if candidates else current_dir


def load_dataset_metadata(dataset_name: str, project_path: str | None = None) -> Dict:
    meta_path = os.path.join(resolve_project_path(project_path), "traindata", f"{dataset_name}_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Dataset metadata not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Preprocessor
# -----------------------------------------------------------------------------

@dataclass
class PreprocessMetadata:
    dataset_name: str
    columns: List[str]
    numeric_cols: List[str]
    categorical_cols: List[str]
    target_col: str
    discrete_col_ids: List[int]
    continuous_col_ids: List[int]
    mu: np.ndarray
    sigma: np.ndarray
    raw_min: np.ndarray
    raw_max: np.ndarray
    transformed_min: np.ndarray
    transformed_max: np.ndarray
    category_maps: Dict[str, Dict[str, int]]
    # Columns that receive log1p transform before standardization (e.g. fnlwgt, capital_gain, capital_loss)
    log1p_col_ids: List[int] = None

    def __post_init__(self):
        if self.log1p_col_ids is None:
            self.log1p_col_ids = []

    def save(self, path: str) -> None:
        np.savez(
            path,
            dataset_name=self.dataset_name,
            columns=np.array(self.columns, dtype=object),
            numeric_cols=np.array(self.numeric_cols, dtype=object),
            categorical_cols=np.array(self.categorical_cols, dtype=object),
            target_col=self.target_col,
            discrete_col_ids=np.array(self.discrete_col_ids, dtype=np.int64),
            continuous_col_ids=np.array(self.continuous_col_ids, dtype=np.int64),
            mu=self.mu.astype(np.float32),
            sigma=self.sigma.astype(np.float32),
            raw_min=self.raw_min.astype(np.float32),
            raw_max=self.raw_max.astype(np.float32),
            transformed_min=self.transformed_min.astype(np.float32),
            transformed_max=self.transformed_max.astype(np.float32),
            category_maps=np.array([json.dumps(self.category_maps)], dtype=object),
            log1p_col_ids=np.array(self.log1p_col_ids, dtype=np.int64),
        )

    @staticmethod
    def load(path: str) -> "PreprocessMetadata":
        payload = np.load(path, allow_pickle=True)
        log1p_col_ids = payload["log1p_col_ids"].astype(np.int64).tolist() if "log1p_col_ids" in payload else []
        return PreprocessMetadata(
            dataset_name=str(payload["dataset_name"]),
            columns=payload["columns"].tolist(),
            numeric_cols=payload["numeric_cols"].tolist(),
            categorical_cols=payload["categorical_cols"].tolist(),
            target_col=str(payload["target_col"]),
            discrete_col_ids=payload["discrete_col_ids"].astype(np.int64).tolist(),
            continuous_col_ids=payload["continuous_col_ids"].astype(np.int64).tolist(),
            mu=payload["mu"].astype(np.float32),
            sigma=payload["sigma"].astype(np.float32),
            raw_min=payload["raw_min"].astype(np.float32),
            raw_max=payload["raw_max"].astype(np.float32),
            transformed_min=payload["transformed_min"].astype(np.float32),
            transformed_max=payload["transformed_max"].astype(np.float32),
            category_maps=json.loads(payload["category_maps"].tolist()[0]),
            log1p_col_ids=log1p_col_ids,
        )


class TabularPreprocessor:
    """Single-source-of-truth preprocessing for training and inference."""

    def __init__(self, metadata: PreprocessMetadata):
        self.meta = metadata
        self.columns = metadata.columns
        self.col_map = {c: i for i, c in enumerate(self.columns)}
        self.discrete_set = set(metadata.discrete_col_ids)

    @classmethod
    def fit_generic(
        cls,
        train_array: np.ndarray,
        *,
        dataset_name: str,
        columns: Sequence[str],
        numeric_cols: Sequence[str],
        categorical_cols: Sequence[str],
        target_col: str = "",
        discrete_cols: Sequence[str] | None = None,
        continuous_cols: Sequence[str] | None = None,
        category_maps: Dict[str, Dict[str, int]] | None = None,
        log1p_cols: Sequence[str] | None = None,
    ) -> "TabularPreprocessor":
        columns = list(columns)
        if train_array.ndim != 2:
            raise ValueError(f"train_array must be 2-D, got shape {train_array.shape}")
        if train_array.shape[1] != len(columns):
            raise ValueError(
                f"Feature count mismatch: train_array has {train_array.shape[1]} columns "
                f"but metadata declares {len(columns)} columns"
            )

        category_maps = dict(category_maps or {})
        numeric_cols = [c for c in numeric_cols if c in columns]
        categorical_cols = [c for c in categorical_cols if c in columns]

        discrete_cols = list(discrete_cols or [])
        if target_col and target_col in columns and target_col not in discrete_cols:
            discrete_cols.append(target_col)
        discrete_cols = [c for c in discrete_cols if c in columns]
        discrete_set = set(discrete_cols)

        if continuous_cols is None:
            continuous_cols = [c for c in columns if c not in discrete_set]
        else:
            continuous_cols = [c for c in continuous_cols if c in columns]
        continuous_set = set(continuous_cols)

        overlap = discrete_set & continuous_set
        if overlap:
            raise ValueError(f"Columns cannot be both discrete and continuous: {sorted(overlap)}")

        discrete_col_ids = [idx for idx, col in enumerate(columns) if col in discrete_set]
        continuous_col_ids = [idx for idx, col in enumerate(columns) if col in continuous_set]

        log1p_cols = [c for c in (log1p_cols or []) if c in columns]
        log1p_col_ids = [columns.index(c) for c in log1p_cols]

        transformed = train_array.astype(np.float32).copy()
        for idx in log1p_col_ids:
            transformed[:, idx] = np.log1p(transformed[:, idx])

        transformed_for_stats = transformed.copy()
        if discrete_col_ids:
            transformed_for_stats[:, discrete_col_ids] += 0.5

        mu = transformed_for_stats.mean(axis=0).astype(np.float32)
        sigma = transformed_for_stats.std(axis=0).astype(np.float32)
        sigma = np.where(sigma < 1e-6, 1.0, sigma).astype(np.float32)

        raw_min = train_array.min(axis=0).astype(np.float32)
        raw_max = train_array.max(axis=0).astype(np.float32)
        transformed_min = np.empty(len(columns), dtype=np.float32)
        transformed_max = np.empty(len(columns), dtype=np.float32)

        transformed_space_min = transformed.min(axis=0).astype(np.float32)
        transformed_space_max = transformed.max(axis=0).astype(np.float32)
        for idx in range(len(columns)):
            transformed_min[idx] = (transformed_space_min[idx] - mu[idx]) / sigma[idx]
            hi = transformed_space_max[idx] + 1.0 if idx in discrete_col_ids else transformed_space_max[idx]
            transformed_max[idx] = (hi - mu[idx]) / sigma[idx]

        meta = PreprocessMetadata(
            dataset_name=dataset_name,
            columns=columns,
            numeric_cols=list(numeric_cols),
            categorical_cols=list(categorical_cols),
            target_col=target_col,
            discrete_col_ids=discrete_col_ids,
            continuous_col_ids=continuous_col_ids,
            mu=mu,
            sigma=sigma,
            raw_min=raw_min,
            raw_max=raw_max,
            transformed_min=transformed_min,
            transformed_max=transformed_max,
            category_maps=category_maps,
            log1p_col_ids=log1p_col_ids,
        )
        return cls(meta)

    def transform_array(self, array: np.ndarray, rng: np.random.RandomState | None = None, add_noise: bool = True) -> np.ndarray:
        array = array.astype(np.float32).copy()
        # Apply log1p to skewed columns before standardization
        if self.meta.log1p_col_ids:
            for idx in self.meta.log1p_col_ids:
                array[:, idx] = np.log1p(array[:, idx])
        if add_noise and self.meta.discrete_col_ids:
            if rng is None:
                rng = np.random.RandomState(1234)
            noise = rng.rand(array.shape[0], len(self.meta.discrete_col_ids)).astype(np.float32)
            array[:, self.meta.discrete_col_ids] += noise
        return ((array - self.meta.mu) / self.meta.sigma).astype(np.float32)

    def transform_value(self, col_id: int, value: Union[str, int, float], value_space: str = "raw") -> float:
        if value_space == "standardized":
            return float(value)

        col_name = self.columns[col_id]
        if isinstance(value, str) and col_name in self.meta.category_maps:
            value = self.meta.category_maps[col_name][value]
        value = float(value)
        return float((value - self.meta.mu[col_id]) / self.meta.sigma[col_id])

    def transform_interval(self, col_id: int, left: float, right: float, value_space: str = "raw") -> Tuple[float, float]:
        if value_space == "standardized":
            return float(left), float(right)
        return (
            float((left - self.meta.mu[col_id]) / self.meta.sigma[col_id]),
            float((right - self.meta.mu[col_id]) / self.meta.sigma[col_id]),
        )

    def discrete_interval_for_value(self, col_id: int, value: Union[str, int, float], value_space: str = "raw") -> Tuple[float, float]:
        if value_space == "standardized":
            z = float(value)
            width = 1.0 / float(self.meta.sigma[col_id])
            return z, z + width

        col_name = self.columns[col_id]
        if isinstance(value, str) and col_name in self.meta.category_maps:
            value = self.meta.category_maps[col_name][value]
        value = float(value)
        return self.transform_interval(col_id, value, value + 1.0, value_space="raw")

__all__ = [
    "ADULT_CONTINUOUS_COLS",
    "ADULT_DISCRETE_NUMERIC_COLS",
    "ADULT_NUMERIC_COLS",
    "ADULT_CATEGORICAL_COLS",
    "ADULT_TARGET_COL",
    "ADULT_ALL_COLS",
    "PreprocessMetadata",
    "TabularPreprocessor",
    "infer_project_path",
    "load_dataset_metadata",
    "resolve_project_path",
]
