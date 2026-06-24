#!/usr/bin/env python
# coding: utf-8

"""Hospital preprocessing for unified flow training."""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    from utils.dataUtils import resolve_project_path
except ImportError:
    from dataUtils import resolve_project_path


HOSPITAL_ALL_COLS = [
    "ProviderNumber",
    "HospitalName",
    "City",
    "State",
    "ZIPCode",
    "CountyName",
    "PhoneNumber",
    "HospitalType",
    "HospitalOwner",
    "EmergencyService",
    "Condition",
    "MeasureCode",
    "MeasureName",
    "Sample",
    "StateAvg",
]

HOSPITAL_CATEGORICAL_COLS = [
    "ProviderNumber",
    "HospitalName",
    "City",
    "State",
    "ZIPCode",
    "CountyName",
    "PhoneNumber",
    "HospitalType",
    "HospitalOwner",
    "EmergencyService",
    "Condition",
    "MeasureCode",
    "MeasureName",
    "StateAvg",
]

HOSPITAL_NUMERIC_COLS = ["Sample"]
HOSPITAL_DISCRETE_NUMERIC_COLS = ["Sample"]
HOSPITAL_CONTINUOUS_COLS: list[str] = []


def parse_sample_count(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    match = re.search(r"-?\d+", text)
    if match is None:
        return np.nan
    return float(match.group())


def preprocess_hospital(
    input_file: str = "hospital.csv",
    output_file: str = "hospital.npy",
    project_path: str | None = None,
) -> Tuple[np.ndarray, list[str]]:
    project_path = resolve_project_path(project_path)
    data_dir = os.path.join(project_path, "traindata")
    os.makedirs(data_dir, exist_ok=True)

    input_file_path = os.path.join(data_dir, input_file)
    df = pd.read_csv(input_file_path, low_memory=False)
    print(f"Original data shape: {df.shape}")
    print(f"Missing value summary:\n{df.isnull().sum()}")

    df = df[HOSPITAL_ALL_COLS].copy()

    label_maps: Dict[str, Dict[str, int]] = {}
    print("\nEncoding categorical columns...")
    for col in HOSPITAL_CATEGORICAL_COLS:
        non_null_mask = df[col].notna()
        series = pd.Series(np.nan, index=df.index, dtype=object)
        series[non_null_mask] = df[col][non_null_mask].astype(str).str.strip()
        unique_values = series.dropna().unique()
        value_to_idx = {value: idx for idx, value in enumerate(unique_values)}
        value_to_idx["Unknown"] = -1  # sentinel for AR inverse map
        encoded = series.map(value_to_idx)
        df[col] = encoded
        label_maps[col] = value_to_idx
        print(f"  {col}: {len(unique_values)} categories")

    df["Sample"] = df["Sample"].apply(parse_sample_count)
    print(f"\nParsed Sample column. Range: [{df['Sample'].min()}, {df['Sample'].max()}]")

    data = df.values.astype(np.float32)
    output_path = os.path.join(data_dir, output_file)
    np.save(output_path, data)

    meta = {
        "all_cols": HOSPITAL_ALL_COLS,
        "categorical_cols": HOSPITAL_CATEGORICAL_COLS,
        "numeric_cols": HOSPITAL_NUMERIC_COLS,
        "discrete_numeric_cols": HOSPITAL_DISCRETE_NUMERIC_COLS,
        "continuous_cols": HOSPITAL_CONTINUOUS_COLS,
        "target_col": "",
        "num_features": int(data.shape[1]),
        "category_maps": label_maps,
        "note": "Categorical columns are label-encoded. Sample is parsed as a discrete patient count and missing values are mapped to -1.",
    }
    meta_path = os.path.join(data_dir, "hospital_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nPreprocessing complete.")
    print(f"Final data shape: {data.shape}")
    print(f"Saved array to: {output_path}")
    print(f"Saved metadata to: {meta_path}")

    print("\nColumn order:")
    for idx, col in enumerate(HOSPITAL_ALL_COLS):
        print(f"  {idx}: {col}")

    return data, HOSPITAL_ALL_COLS


if __name__ == "__main__":
    preprocess_hospital("hospital.csv", "hospital.npy")
