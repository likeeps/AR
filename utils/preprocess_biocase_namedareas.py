#!/usr/bin/env python
# coding: utf-8

"""BioCase NamedAreas (r137711_c11) preprocessing for AR pipeline.

Dataset: t_biocase_gathering_namedareas_r137711_c11 (137710 rows, 11 columns)
Ground truth FDs:
  AreaName -> AreaClass, AreaName -> AreaCode,
  AreaCode -> AreaName, AreaCode -> AreaClass,
  _unitguid -> _datasetguid

Columns kept (6):
  _datasetguid     - 3 values       → categorical
  _unitguid        - 76466 unique   → identifier (categorical)
  Sequence         - 2 values       → discrete numeric
  AreaClass        - 2 values       → categorical
  AreaCode         - 641 unique, 37.9% null → categorical
  AreaName         - 1722 unique    → categorical

Columns dropped:
  _namedareasguid      - unique per row (identifier)
  AreaClass_Language   - constant (1 value)
  AreaCodeStandard     - 100% null
  AreaName_Language    - constant (1 value)
  DataSource           - 100% null
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
    "_datasetguid",
    "_unitguid",
    "Sequence",
    "AreaClass",
    "AreaCode",
    "AreaName",
]

CATEGORICAL_COLS = [
    "_datasetguid",
    "_unitguid",
    "AreaClass",
    "AreaCode",
    "AreaName",
]

DISCRETE_NUMERIC_COLS = ["Sequence"]
NUMERIC_COLS = ["Sequence"]
CONTINUOUS_COLS: list[str] = []

DROPPED_COLUMNS = [
    "_namedareasguid",
    "AreaClass_Language",
    "AreaCodeStandard",
    "AreaName_Language",
    "DataSource",
]


def preprocess_biocase_namedareas(
    input_file: str = "t_biocase_gathering_namedareas_r137711_c11.csv",
    output_file: str = "biocase_namedareas.npy",
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

    # Discrete numeric columns
    print("\nDiscrete numeric columns...")
    for col in DISCRETE_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"  {col}: range [{df[col].min()}, {df[col].max()}]")

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
            "BioCase namedareas data (r137711, 6 effective columns). "
            "Ground truth FDs: AreaName->AreaClass, AreaName->AreaCode, "
            "AreaCode->AreaName, AreaCode->AreaClass, _unitguid->_datasetguid."
        ),
    }
    meta_path = os.path.join(data_dir, "biocase_namedareas_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Shape: {data.shape}, saved to {output_path}")
    return data, ALL_COLS


if __name__ == "__main__":
    preprocess_biocase_namedareas()
