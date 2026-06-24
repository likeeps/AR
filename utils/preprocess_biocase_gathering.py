#!/usr/bin/env python
# coding: utf-8

"""BioCase Gathering (r90992_c35) preprocessing for AR pipeline.

Dataset: t_biocase_gathering_r90992_c35 (90991 rows, 35 columns)
Ground truth FD: Gath_AreaDetail -> Gath_Country_Name

Columns kept (6):
  _datasetguid          - 3 values       → categorical
  _unitguid             - unique ID      → identifier (categorical)
  Gath_AreaDetail       - 8833 values, 10.6% null → high-cardinality categorical
  Gath_Country_Name     - 310 values, 16.4% null  → categorical
  Gath_DateTime_Begin   - 16876 values, 41.6% null → categorical
  Gath_DateTime_End     - 16608 values, 43.4% null → categorical

Columns dropped:
  20 columns at 100% null (Gath_Altid_*, Gath_Code, Gath_Country_NameDerived*,
    Gath_Country_ISO, Gath_DateTime_TimeZone, Gath_Depth_*, Gath_GML,
    Gath_LocalityText, Gath_Method)
  3 constant columns (Gath_Altid_Parameter, Gath_Country_Name_Language,
    Gath_Depth_Parameter)
  Gath_Notes - 99.9% null (25 unique, too sparse)
  Gath_DateTime_Text - 84.1% null (redundant with Begin/End)
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


BIOCASE_GATH_ALL_COLS = [
    "_datasetguid",
    "_unitguid",
    "Gath_AreaDetail",
    "Gath_Country_Name",
    "Gath_DateTime_Begin",
    "Gath_DateTime_End",
]

BIOCASE_GATH_CATEGORICAL_COLS = [
    "_datasetguid",
    "_unitguid",
    "Gath_AreaDetail",
    "Gath_Country_Name",
    "Gath_DateTime_Begin",
    "Gath_DateTime_End",
]

BIOCASE_GATH_DISCRETE_NUMERIC_COLS: list[str] = []
BIOCASE_GATH_NUMERIC_COLS: list[str] = []
BIOCASE_GATH_CONTINUOUS_COLS: list[str] = []

BIOCASE_GATH_DROPPED_COLUMNS = [
    "Gath_Altid_Accuracy",
    "Gath_Altid_IsQuantitative",
    "Gath_Altid_LowerValue",
    "Gath_Altid_DateTime",
    "Gath_Altid_Method",
    "Gath_Altid_Parameter",
    "Gath_Altid_UnitOfMeasurement",
    "Gath_Altid_UpperValue",
    "Gath_Altid_MOFText",
    "Gath_Code",
    "Gath_Country_Name_Language",
    "Gath_Country_NameDerived",
    "Gath_Country_NameDerived_Language",
    "Gath_Country_ISO",
    "Gath_DateTime_Text",
    "Gath_DateTime_TimeZone",
    "Gath_Depth_Accuracy",
    "Gath_Depth_IsQuantitative",
    "Gath_Depth_LowerValue",
    "Gath_Depth_DateTime",
    "Gath_Depth_Method",
    "Gath_Depth_Parameter",
    "Gath_Depth_UnitOfMeasurement",
    "Gath_Depth_UpperValue",
    "Gath_Depth_MOFText",
    "Gath_GML",
    "Gath_LocalityText",
    "Gath_Method",
    "Gath_Notes",
]


def preprocess_biocase_gathering(
    input_file: str = "t_biocase_gathering_r90992_c35.csv",
    output_file: str = "biocase_gathering.npy",
    project_path: str | None = None,
) -> Tuple[np.ndarray, list[str]]:
    project_path = resolve_project_path(project_path)
    data_dir = os.path.join(project_path, "traindata")
    os.makedirs(data_dir, exist_ok=True)

    input_file_path = os.path.join(data_dir, input_file)
    df = pd.read_csv(input_file_path, low_memory=False)
    print(f"Original data shape: {df.shape}")

    # Select effective columns
    df = df[BIOCASE_GATH_ALL_COLS].copy()

    # Encode categorical columns (NaN-preserving)
    label_maps: Dict[str, Dict[str, int]] = {}
    print("\nEncoding categorical columns...")
    for col in BIOCASE_GATH_CATEGORICAL_COLS:
        non_null_mask = df[col].notna()
        series = pd.Series(np.nan, index=df.index, dtype=object)
        series[non_null_mask] = df[col][non_null_mask].astype(str).str.strip()
        unique_values = series.dropna().unique()
        value_to_idx = {value: idx for idx, value in enumerate(unique_values)}
        value_to_idx["Unknown"] = -1  # sentinel for AR inverse map
        encoded = series.map(value_to_idx)
        df[col] = encoded
        label_maps[col] = value_to_idx
        null_count = df[col].isnull().sum()
        print(f"  {col}: {len(unique_values)} categories, {null_count} nulls")

    data = df.values.astype(np.float32)
    output_path = os.path.join(data_dir, output_file)
    np.save(output_path, data)

    meta = {
        "all_cols": BIOCASE_GATH_ALL_COLS,
        "categorical_cols": BIOCASE_GATH_CATEGORICAL_COLS,
        "numeric_cols": BIOCASE_GATH_NUMERIC_COLS,
        "continuous_cols": BIOCASE_GATH_CONTINUOUS_COLS,
        "discrete_numeric_cols": BIOCASE_GATH_DISCRETE_NUMERIC_COLS,
        "target_col": "",
        "dropped_columns": BIOCASE_GATH_DROPPED_COLUMNS,
        "num_features": int(data.shape[1]),
        "category_maps": label_maps,
        "note": (
            "BioCase gathering data (r90992, 35 raw columns, 6 effective). "
            "Ground truth FD: Gath_AreaDetail -> Gath_Country_Name. "
            "20 columns at 100% null dropped, 3 constant columns dropped, "
            "Gath_Notes (99.9% null) and Gath_DateTime_Text (84.1% null) dropped."
        ),
    }
    meta_path = os.path.join(data_dir, "biocase_gathering_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Shape: {data.shape}, saved to {output_path}")
    print(f"Metadata: {meta_path}")

    print("\nColumn order:")
    for i, col in enumerate(BIOCASE_GATH_ALL_COLS):
        print(f"  {i}: {col}")

    return data, BIOCASE_GATH_ALL_COLS


if __name__ == "__main__":
    preprocess_biocase_gathering()
