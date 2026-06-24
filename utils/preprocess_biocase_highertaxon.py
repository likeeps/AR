#!/usr/bin/env python
# coding: utf-8

"""BioCase Identification HigherTaxon (r562959_c3) preprocessing for AR pipeline.

Dataset: t_biocase_identification_highertaxon_r562959_c3 (562958 rows, 3 columns)
Ground truth FD: HigherTaxonName -> HigherTaxonRank

Columns (all kept):
  _identificationguid - 91799 unique → identifier (categorical)
  HigherTaxonName     - 3660 unique  → categorical
  HigherTaxonRank     - 9 unique     → categorical
"""

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


ALL_COLS = [
    "_identificationguid",
    "HigherTaxonName",
    "HigherTaxonRank",
]

CATEGORICAL_COLS = [
    "_identificationguid",
    "HigherTaxonName",
    "HigherTaxonRank",
]

DISCRETE_NUMERIC_COLS: list[str] = []
NUMERIC_COLS: list[str] = []
CONTINUOUS_COLS: list[str] = []

DROPPED_COLUMNS: list[str] = []


def preprocess_biocase_highertaxon(
    input_file: str = "t_biocase_identification_highertaxon_r562959_c3.csv",
    output_file: str = "biocase_highertaxon.npy",
    project_path: str | None = None,
) -> Tuple[np.ndarray, list[str]]:
    project_path = resolve_project_path(project_path)
    data_dir = os.path.join(project_path, "traindata")
    os.makedirs(data_dir, exist_ok=True)

    input_file_path = os.path.join(data_dir, input_file)
    df = pd.read_csv(input_file_path, low_memory=False)
    print(f"Original data shape: {df.shape}")

    df = df[ALL_COLS].copy()

    # Encode categorical columns (NaN-preserving)
    label_maps: Dict[str, Dict[str, int]] = {}
    print("\nEncoding categorical columns...")
    for col in CATEGORICAL_COLS:
        non_null_mask = df[col].notna()
        series = pd.Series(np.nan, index=df.index, dtype=object)
        series[non_null_mask] = df[col][non_null_mask].astype(str).str.strip()
        unique_values = series.dropna().unique()
        value_to_idx = {value: idx for idx, value in enumerate(unique_values)}
        value_to_idx["Unknown"] = -1
        encoded = series.map(value_to_idx)
        df[col] = encoded
        label_maps[col] = value_to_idx
        null_count = df[col].isnull().sum()
        print(f"  {col}: {len(unique_values)} categories, {null_count} nulls")

    data = df.values.astype(np.float32)
    output_path = os.path.join(data_dir, output_file)
    np.save(output_path, data)

    meta = {
        "all_cols": ALL_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
        "continuous_cols": CONTINUOUS_COLS,
        "discrete_numeric_cols": DISCRETE_NUMERIC_COLS,
        "target_col": "",
        "dropped_columns": DROPPED_COLUMNS,
        "num_features": int(data.shape[1]),
        "category_maps": label_maps,
        "note": (
            "BioCase identification highertaxon data (r562959, 3 columns). "
            "Ground truth FD: HigherTaxonName -> HigherTaxonRank."
        ),
    }
    meta_path = os.path.join(data_dir, "biocase_highertaxon_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Shape: {data.shape}, saved to {output_path}")
    return data, ALL_COLS


if __name__ == "__main__":
    preprocess_biocase_highertaxon()
