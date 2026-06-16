#!/usr/bin/env python
# coding: utf-8

"""Generic preprocessor for datasets that follow the standard pattern.

This replaces the boilerplate in individual preprocess_*.py files.
Custom datasets (adult, claims, hospital, dblp10k) keep their own preprocessors.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from utils.dataUtils import resolve_project_path
except ImportError:
    from dataUtils import resolve_project_path


@dataclass
class PreprocessConfig:
    """Declarative config for a standard dataset."""
    dataset_name: str
    input_csv: str
    output_npy: str
    meta_json: str
    all_cols: list[str]
    categorical_cols: list[str]
    numeric_cols: list[str] = field(default_factory=list)
    continuous_cols: list[str] = field(default_factory=list)
    discrete_numeric_cols: list[str] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    target_col: str = ""
    note: str = ""


def preprocess_generic(
    config: PreprocessConfig,
    project_path: str | None = None,
) -> Tuple[np.ndarray, list[str]]:
    """Standard preprocessing: label-encode categoricals, passthrough numerics."""
    project_path = resolve_project_path(project_path)
    data_dir = os.path.join(project_path, "traindata")
    os.makedirs(data_dir, exist_ok=True)

    input_path = os.path.join(data_dir, config.input_csv)
    df = pd.read_csv(input_path, low_memory=False)
    print(f"[{config.dataset_name}] Original shape: {df.shape}")

    df = df[config.all_cols].copy()

    # Label-encode categorical columns (NaN-preserving)
    label_maps: Dict[str, Dict[str, int]] = {}
    all_numeric = set(config.numeric_cols) | set(config.continuous_cols) | set(config.discrete_numeric_cols)
    print(f"\n[{config.dataset_name}] Encoding categorical columns...")
    for col in config.categorical_cols:
        if col in all_numeric:
            continue
        non_null_mask = df[col].notna()
        series = pd.Series(np.nan, index=df.index, dtype=object)
        series[non_null_mask] = df[col][non_null_mask].astype(str).str.strip()
        unique_values = series.dropna().unique()
        value_to_idx = {value: idx for idx, value in enumerate(unique_values)}
        value_to_idx["Unknown"] = -1
        encoded = series.map(value_to_idx)
        df[col] = encoded
        label_maps[col] = value_to_idx
        null_count = int(df[col].isnull().sum())
        print(f"  {col}: {len(unique_values)} categories, {null_count} nulls")

    # Parse numeric columns (preserve NaN)
    for col in all_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    data = df.values.astype(np.float32)
    output_path = os.path.join(data_dir, config.output_npy)
    np.save(output_path, data)

    meta = {
        "all_cols": config.all_cols,
        "categorical_cols": config.categorical_cols,
        "numeric_cols": config.numeric_cols,
        "continuous_cols": config.continuous_cols,
        "discrete_numeric_cols": config.discrete_numeric_cols,
        "target_col": config.target_col,
        "dropped_columns": config.dropped_columns,
        "num_features": int(data.shape[1]),
        "category_maps": label_maps,
        "note": config.note,
    }
    meta_path = os.path.join(data_dir, config.meta_json)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[{config.dataset_name}] Done. Shape: {data.shape}, saved to {output_path}")
    return data, config.all_cols
