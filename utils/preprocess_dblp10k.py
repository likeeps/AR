#!/usr/bin/env python
# coding: utf-8

"""DBLP10K preprocessing for unified flow training."""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    from utils.dataUtils import resolve_project_path
except ImportError:
    from dataUtils import resolve_project_path


DBLP10K_DROPPED_COLS = [
    "p1editor",
    "p1address",
    "p2editor",
    "p2address",
]

DBLP10K_ALL_COLS = [
    "sameentity",
    "samename",
    "author1",
    "author2",
    "key1",
    "key2",
    "p1type",
    "p1author",
    "p1title",
    "p1booktitle",
    "p1booktitlefull",
    "p1year",
    "p1journal",
    "p1journalfull",
    "p1publisher",
    "p1series",
    "p1id",
    "p1key",
    "p2type",
    "p2author",
    "p2title",
    "p2booktitle",
    "p2booktitlefull",
    "p2year",
    "p2journal",
    "p2journalfull",
    "p2publisher",
    "p2series",
    "p2id",
    "p2key",
]

DBLP10K_CATEGORICAL_COLS = [
    "sameentity",
    "samename",
    "author1",
    "author2",
    "key1",
    "key2",
    "p1type",
    "p1author",
    "p1title",
    "p1booktitle",
    "p1booktitlefull",
    "p1journal",
    "p1journalfull",
    "p1publisher",
    "p1series",
    "p1id",
    "p1key",
    "p2type",
    "p2author",
    "p2title",
    "p2booktitle",
    "p2booktitlefull",
    "p2journal",
    "p2journalfull",
    "p2publisher",
    "p2series",
    "p2id",
    "p2key",
]

DBLP10K_NUMERIC_COLS = ["p1year", "p2year"]
DBLP10K_DISCRETE_NUMERIC_COLS = ["p1year", "p2year"]
DBLP10K_CONTINUOUS_COLS: list[str] = []

_BOOLEAN_CATEGORY_MAP = {
    "f": 0,
    "false": 0,
    "0": 0,
    "t": 1,
    "true": 1,
    "1": 1,
}


def _normalize_text_series(series: pd.Series) -> pd.Series:
    non_null_mask = series.notna()
    normalized = pd.Series(np.nan, index=series.index, dtype=object)
    normalized[non_null_mask] = series[non_null_mask].astype(str).str.strip()
    # Treat empty strings and string representations of null as actual null
    null_strings = {"", "nan", "none"}
    for idx in normalized[non_null_mask].index:
        if normalized[idx].lower() in null_strings:
            normalized.at[idx] = np.nan
    return normalized


def preprocess_dblp10k(
    input_file: str = "dblp10k.csv",
    output_file: str = "dblp10k.npy",
    project_path: str | None = None,
) -> Tuple[np.ndarray, list[str]]:
    project_path = resolve_project_path(project_path)
    data_dir = os.path.join(project_path, "traindata")
    os.makedirs(data_dir, exist_ok=True)

    input_file_path = os.path.join(data_dir, input_file)
    df = pd.read_csv(input_file_path, low_memory=False)
    print(f"Original data shape: {df.shape}")
    print(f"Missing value summary:\n{df.isnull().sum()}")

    df = df[DBLP10K_ALL_COLS].copy()

    label_maps: Dict[str, Dict[str, int]] = {}
    print("\nEncoding categorical columns...")
    for col in DBLP10K_CATEGORICAL_COLS:
        normalized = _normalize_text_series(df[col])
        non_null_values = normalized.dropna()
        unique_values = non_null_values.unique().tolist()
        normalized_lower = {value.lower() for value in unique_values}

        if normalized_lower and normalized_lower.issubset(set(_BOOLEAN_CATEGORY_MAP.keys())):
            value_to_idx = {
                value: _BOOLEAN_CATEGORY_MAP[value.lower()]
                for value in unique_values
            }
        else:
            value_to_idx = {value: idx for idx, value in enumerate(unique_values)}

        value_to_idx["Unknown"] = -1  # sentinel for AR inverse map
        encoded = normalized.map(value_to_idx)
        df[col] = encoded
        label_maps[col] = value_to_idx
        print(f"  {col}: {len(unique_values)} categories")

    print("\nParsing discrete numeric year columns...")
    for col in DBLP10K_DISCRETE_NUMERIC_COLS:
        numeric = pd.to_numeric(df[col], errors="coerce")
        df[col] = numeric
        print(f"  {col}: range [{numeric.min()}, {numeric.max()}]")

    data = df.values.astype(np.float32)
    output_path = os.path.join(data_dir, output_file)
    np.save(output_path, data)

    meta = {
        "all_cols": DBLP10K_ALL_COLS,
        "categorical_cols": DBLP10K_CATEGORICAL_COLS,
        "numeric_cols": DBLP10K_NUMERIC_COLS,
        "discrete_numeric_cols": DBLP10K_DISCRETE_NUMERIC_COLS,
        "continuous_cols": DBLP10K_CONTINUOUS_COLS,
        "target_col": "",
        "num_features": int(data.shape[1]),
        "category_maps": label_maps,
        "dropped_columns": DBLP10K_DROPPED_COLS,
        "note": (
            "DBLP10K keeps identifier-like columns such as p1id/p2id as categorical codes so "
            "they remain exact-match searchable. Four fully-missing columns are dropped."
        ),
    }
    meta_path = os.path.join(data_dir, "dblp10k_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nPreprocessing complete.")
    print(f"Final data shape: {data.shape}")
    print(f"Saved array to: {output_path}")
    print(f"Saved metadata to: {meta_path}")
    print(f"Dropped columns: {DBLP10K_DROPPED_COLS}")

    print("\nColumn order:")
    for idx, col in enumerate(DBLP10K_ALL_COLS):
        print(f"  {idx}: {col}")

    return data, DBLP10K_ALL_COLS


if __name__ == "__main__":
    preprocess_dblp10k("dblp10k.csv", "dblp10k.npy")
