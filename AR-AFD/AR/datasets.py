from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any

import numpy as np

from AR.column_policy import ColumnPolicy, build_column_policy_map
from utils.dataUtils import load_dataset_metadata


@lru_cache(maxsize=1)
def _load_dataset_registry_module(repo_root: str):
    registry_path = Path(repo_root) / "train" / "dataset_registry.py"
    if not registry_path.exists():
        raise FileNotFoundError(f"Dataset registry not found: {registry_path}")

    spec = importlib.util.spec_from_file_location("face_c_dataset_registry", registry_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load dataset registry from: {registry_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize_metadata(raw_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "all_cols": list(raw_meta.get("all_cols") or raw_meta.get("columns") or []),
        "numeric_cols": list(raw_meta.get("numeric_cols", raw_meta.get("numerical_cols", []))),
        "categorical_cols": list(raw_meta.get("categorical_cols", [])),
        "discrete_numeric_cols": list(raw_meta.get("discrete_numeric_cols", [])),
        "continuous_cols": list(raw_meta.get("continuous_cols", [])),
        "date_cols": list(raw_meta.get("date_cols", [])),
        "dropped_columns": list(raw_meta.get("dropped_columns", [])),
        "target_col": str(raw_meta.get("target_col", "")),
        "category_maps": dict(raw_meta.get("category_maps", {})),
    }


def _invert_category_maps(
    category_maps: dict[str, dict[str, Any]],
) -> dict[str, dict[int, str]]:
    inverse: dict[str, dict[int, str]] = {}
    for column_name, mapping in category_maps.items():
        inverse[column_name] = {
            int(encoded): str(raw_value)
            for raw_value, encoded in dict(mapping).items()
        }
    return inverse


def _compute_column_stats(selected_array: np.ndarray) -> dict[int, dict[str, float]]:
    row_count = max(int(selected_array.shape[0]), 1)
    stats: dict[int, dict[str, float]] = {}
    for col_id in range(selected_array.shape[1]):
        col = selected_array[:, col_id]
        unique_count = int(np.unique(col).size)
        # marginal_top1: frequency of the most common value (high = near-constant)
        values, counts = np.unique(col, return_counts=True)
        marginal_top1 = float(counts.max() / row_count) if len(counts) > 0 else 1.0
        stats[col_id] = {
            "unique_count": float(unique_count),
            "unique_ratio": float(unique_count / row_count),
            "marginal_top1": marginal_top1,
        }
    return stats


def detect_degenerate_columns(
    column_names: list[str],
    column_stats: dict[int, dict[str, float]],
    *,
    max_top1_ratio: float = 0.95,
    max_unique_count: int = 2,
) -> list[str]:
    """Identify columns that are near-constant or trivially predictable.

    Degenerate columns produce spurious FDs because any column can
    "predict" a near-constant target with high accuracy.

    Returns list of degenerate column names (for logging/filtering).
    """
    degenerate: list[str] = []
    for col_id, name in enumerate(column_names):
        stats = column_stats.get(col_id, {})
        top1 = stats.get("marginal_top1", 0.0)
        unique = stats.get("unique_count", 0.0)
        if top1 >= max_top1_ratio:
            degenerate.append(f"{name} (top1={top1:.3f}, {int(unique)} unique)")
        elif unique <= max_unique_count and unique > 0:
            degenerate.append(f"{name} ({int(unique)} unique values)")
    return degenerate


@dataclass
class DatasetRuntime:
    dataset_name: str
    raw_meta: dict[str, Any]
    metadata: dict[str, Any]
    training_schema: Any
    selected_array: np.ndarray
    selected_indices: tuple[int, ...]
    category_maps: dict[str, dict[str, Any]]
    inverse_category_maps: dict[str, dict[int, str]]
    column_stats: dict[int, dict[str, float]]
    column_policies: dict[int, ColumnPolicy]
    source_total_rows: int = 0
    degenerate_columns: tuple[str, ...] = ()

    @property
    def columns(self) -> list[str]:
        return list(self.training_schema.columns)

    @property
    def target_col(self) -> str:
        return str(self.training_schema.target_col or "")

    @property
    def discrete_col_names(self) -> tuple[str, ...]:
        return tuple(self.training_schema.discrete_cols)

    @property
    def continuous_col_names(self) -> tuple[str, ...]:
        return tuple(self.training_schema.continuous_cols)

    @property
    def categorical_col_names(self) -> tuple[str, ...]:
        return tuple(self.training_schema.categorical_cols)

    @property
    def date_col_names(self) -> tuple[str, ...]:
        return tuple(self.metadata.get("date_cols", []))

    def is_categorical(self, column_name: str) -> bool:
        return column_name in self.categorical_col_names or column_name in self.inverse_category_maps

    def is_continuous(self, column_name: str) -> bool:
        return column_name in self.continuous_col_names

    def is_discrete(self, column_name: str) -> bool:
        return column_name in self.discrete_col_names

    def is_target(self, column_name: str) -> bool:
        return bool(self.target_col) and column_name == self.target_col


def load_runtime_dataset(
    dataset_name: str,
    repo_root: str | Path,
    *,
    sample_rows: int | None = None,
) -> DatasetRuntime:
    repo_root_path = Path(repo_root)
    dataset_registry = _load_dataset_registry_module(str(repo_root_path))
    spec = dataset_registry.get_dataset_spec(dataset_name)
    raw_meta = load_dataset_metadata(spec.dataset_name, project_path=str(repo_root_path))
    metadata = normalize_metadata(raw_meta)

    raw_columns = list(metadata["all_cols"])
    if not raw_columns:
        raise ValueError(f"Dataset {dataset_name} metadata does not declare all_cols/columns.")

    source_path = repo_root_path / "traindata" / f"{spec.dataset_name}.npy"
    if not source_path.exists():
        raise FileNotFoundError(f"Processed dataset array not found: {source_path}")

    raw_array = np.load(source_path, mmap_mode="r")
    if raw_array.ndim != 2:
        raise ValueError(f"Expected 2-D array for {dataset_name}, got shape {raw_array.shape}")
    if raw_array.shape[1] != len(raw_columns):
        raise ValueError(
            f"Column mismatch for {dataset_name}: array has {raw_array.shape[1]} columns, "
            f"metadata declares {len(raw_columns)} columns."
        )

    training_schema = spec.schema_builder(raw_meta)
    source_total_rows = int(raw_array.shape[0])
    missing_columns = [column for column in training_schema.columns if column not in raw_columns]
    if missing_columns:
        raise KeyError(f"Training schema columns missing from raw metadata for {dataset_name}: {missing_columns}")

    selected_indices = tuple(raw_columns.index(column) for column in training_schema.columns)
    if sample_rows is None:
        selected_array = np.asarray(raw_array[:, selected_indices])
    else:
        selected_array = np.asarray(raw_array[:sample_rows, selected_indices])

    category_maps = {
        column_name: dict(metadata["category_maps"].get(column_name, {}))
        for column_name in training_schema.columns
        if column_name in metadata["category_maps"]
    }
    inverse_category_maps = _invert_category_maps(category_maps)

    discrete_ids = {
        col_id
        for col_id, column_name in enumerate(training_schema.columns)
        if column_name in set(training_schema.discrete_cols)
    }
    column_stats = _compute_column_stats(selected_array)
    column_policies = build_column_policy_map(
        schema=list(training_schema.columns),
        discrete_ids=discrete_ids,
        dataset_metadata=metadata,
        column_stats=column_stats,
    )

    degenerate = detect_degenerate_columns(
        list(training_schema.columns), column_stats
    )
    if degenerate:
        import logging
        logging.getLogger(__name__).warning(
            "Degenerate columns detected (may produce spurious FDs): %s",
            degenerate,
        )

    return DatasetRuntime(
        dataset_name=spec.dataset_name,
        raw_meta=raw_meta,
        metadata=metadata,
        training_schema=training_schema,
        selected_array=selected_array,
        selected_indices=selected_indices,
        category_maps=category_maps,
        inverse_category_maps=inverse_category_maps,
        column_stats=column_stats,
        column_policies=column_policies,
        source_total_rows=source_total_rows,
        degenerate_columns=tuple(degenerate),
    )
