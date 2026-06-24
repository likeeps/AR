#!/usr/bin/env python
# coding: utf-8

"""BioCase Gathering Agent dataset preprocessing for unified flow training.

Dataset: t_biocase_gathering_agent_r72738_c18 (72737 rows, 18 columns)
Columns:
  _datasetguid            - 2 values       → categorical
  _unitguid               - near-unique ID → identifier (dropped)
  _gatheringagentguid     - unique ID      → identifier (dropped)
  AgentText               - 80 values, 99.7% null → categorical
  OrgNameRepresentText    - 100% null      → dropped
  OrgNameRepresent_Language - 100% null    → dropped
  OrgNameRepresentAbbr    - 100% null      → dropped
  OrgUnit                 - 100% null      → dropped
  OrgUnit_Language        - 100% null      → dropped
  PersonFullName          - 2948 values    → categorical
  PersonSortingName       - 100% null      → dropped
  PersonInheritedName     - 2416 values    → categorical
  PersonPrefix            - 100% null      → dropped
  PersonSuffix            - 100% null      → dropped
  PersonGivenNames        - 881 values     → categorical
  PersonPreferredName     - 100% null      → dropped
  PrimaryCollector        - constant (1)   → dropped
  Sequence                - 2 values       → discrete numeric
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


# Columns kept for AR pipeline (3 effective columns).
# Dropped: identifiers, 100%-null columns, constant columns,
# and near-constant columns (_datasetguid: 2 values, AgentText: 99.7% null,
# Sequence: 2 values) which produce trivial spurious FDs.
BIOCASE_ALL_COLS = [
    "_unitguid",
    "_datasetguid",
    "AgentText",
    "PersonFullName",
    "PersonInheritedName",
    "PersonGivenNames",
]

BIOCASE_CATEGORICAL_COLS = [
    "_unitguid",
    "_datasetguid",
    "AgentText",
    "PersonFullName",
    "PersonInheritedName",
    "PersonGivenNames",
]

BIOCASE_DISCRETE_NUMERIC_COLS: list[str] = []
BIOCASE_NUMERIC_COLS: list[str] = []
BIOCASE_CONTINUOUS_COLS: list[str] = []

# Columns dropped: identifiers (_unitguid, _gatheringagentguid),
# 100%-null columns (OrgNameRepresent*, OrgUnit*, PersonSortingName,
# PersonPrefix, PersonSuffix, PersonPreferredName),
# constant column (PrimaryCollector),
# near-constant columns (_datasetguid: 2 values, AgentText: 99.7% null, Sequence: 2 values)
BIOCASE_DROPPED_COLUMNS = [
    "_gatheringagentguid",
    "Sequence",
    "OrgNameRepresentText",
    "OrgNameRepresent_Language",
    "OrgNameRepresentAbbr",
    "OrgUnit",
    "OrgUnit_Language",
    "PersonSortingName",
    "PersonPrefix",
    "PersonSuffix",
    "PersonPreferredName",
    "PrimaryCollector",
]


def preprocess_biocase(
    input_file: str = "t_biocase_gathering_agent_r72738_c18.csv",
    output_file: str = "biocase.npy",
    project_path: str | None = None,
) -> Tuple[np.ndarray, list[str]]:
    project_path = resolve_project_path(project_path)
    data_dir = os.path.join(project_path, "traindata")
    os.makedirs(data_dir, exist_ok=True)

    input_file_path = os.path.join(data_dir, input_file)
    df = pd.read_csv(input_file_path, low_memory=False)
    print(f"原始数据形状: {df.shape}")

    # Select effective columns
    df = df[BIOCASE_ALL_COLS].copy()

    # Encode categorical columns
    label_maps: Dict[str, Dict[str, int]] = {}
    print("\n开始编码分类变量...")
    for col in BIOCASE_CATEGORICAL_COLS:
        non_null_mask = df[col].notna()
        series = pd.Series(np.nan, index=df.index, dtype=object)
        series[non_null_mask] = df[col][non_null_mask].astype(str).str.strip()
        unique_values = series.dropna().unique()
        value_to_idx = {value: idx for idx, value in enumerate(unique_values)}
        value_to_idx["Unknown"] = -1  # sentinel for AR inverse map
        encoded = series.map(value_to_idx)
        df[col] = encoded
        label_maps[col] = value_to_idx
        print(f"  {col}: {len(unique_values)} 个类别")

    # Parse discrete numeric columns
    print("\n解析离散数值列...")
    for col in BIOCASE_DISCRETE_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int64)
        print(f"  {col}: 范围 [{df[col].min()}, {df[col].max()}]")

    data = df.values.astype(np.float32)
    output_path = os.path.join(data_dir, output_file)
    np.save(output_path, data)

    meta = {
        "all_cols": BIOCASE_ALL_COLS,
        "categorical_cols": BIOCASE_CATEGORICAL_COLS,
        "numeric_cols": BIOCASE_NUMERIC_COLS,
        "continuous_cols": BIOCASE_CONTINUOUS_COLS,
        "discrete_numeric_cols": BIOCASE_DISCRETE_NUMERIC_COLS,
        "target_col": "",
        "dropped_columns": BIOCASE_DROPPED_COLUMNS,
        "num_features": int(data.shape[1]),
        "category_maps": label_maps,
        "note": (
            "BioCase gathering agent data. 6 effective columns after dropping "
            "identifiers, 100%-null columns, and constant columns. "
            "PersonFullName/PersonInheritedName are high-cardinality categorical. "
            "Sequence is discrete numeric (2 values)."
        ),
    }
    meta_path = os.path.join(data_dir, "biocase_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n处理完成！")
    print(f"最终数据形状: {data.shape}")
    print(f"特征维度: {data.shape[1]}")
    print(f"样本数量: {data.shape[0]}")
    print(f"数据已保存至: {output_path}")

    print("\n列名与索引对应关系:")
    for i, col in enumerate(BIOCASE_ALL_COLS):
        print(f"  {i}: {col}")

    return data, BIOCASE_ALL_COLS


if __name__ == "__main__":
    preprocess_biocase()
