#!/usr/bin/env python
# coding: utf-8

"""Tax dataset preprocessing for unified flow training.

Dataset: Tax (individual tax records, 15 columns)
Columns:
  FName, LName       - personal names → categorical
  Gender             - M/F            → categorical
  AreaCode           - 3-digit code   → categorical (identifier-like)
  Phone              - phone number   → categorical (identifier-like)
  City, State, Zip   - location info  → categorical
  MaritalStatus      - M/S            → categorical
  HasChild           - Y/N            → categorical
  Salary             - USD amount     → continuous
  Rate               - tax rate (%)   → continuous
  SingleExemp, MarriedExemp, ChildExemp - exemption amounts → discrete numeric
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


TAX_ALL_COLS = [
    "FName",
    "LName",
    "Gender",
    "AreaCode",
    "Phone",
    "City",
    "State",
    "Zip",
    "MaritalStatus",
    "HasChild",
    "Salary",
    "Rate",
    "SingleExemp",
    "MarriedExemp",
    "ChildExemp",
]

TAX_CATEGORICAL_COLS = [
    "FName",
    "LName",
    "Gender",
    "AreaCode",
    "Phone",
    "City",
    "State",
    "Zip",
    "MaritalStatus",
    "HasChild",
]

TAX_CONTINUOUS_COLS = [
    "Salary",
    "Rate",
]

TAX_DISCRETE_NUMERIC_COLS = [
    "SingleExemp",
    "MarriedExemp",
    "ChildExemp",
]

TAX_NUMERIC_COLS = TAX_CONTINUOUS_COLS + TAX_DISCRETE_NUMERIC_COLS


def preprocess_tax(
    input_file: str = "tax.csv",
    output_file: str = "tax.npy",
    project_path: str | None = None,
) -> Tuple[np.ndarray, list[str]]:
    project_path = resolve_project_path(project_path)
    data_dir = os.path.join(project_path, "traindata")
    os.makedirs(data_dir, exist_ok=True)

    input_file_path = os.path.join(data_dir, input_file)
    df = pd.read_csv(input_file_path, low_memory=False)
    print(f"原始数据形状: {df.shape}")
    print(f"缺失值统计:\n{df.isnull().sum()}")

    df = df[TAX_ALL_COLS].copy()

    # Encode categorical columns
    label_maps: Dict[str, Dict[str, int]] = {}
    print("\n开始编码分类变量...")
    for col in TAX_CATEGORICAL_COLS:
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

    # Parse continuous numeric columns
    print("\n解析连续数值列...")
    for col in TAX_CONTINUOUS_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"  {col}: 范围 [{df[col].min():.4f}, {df[col].max():.4f}]")

    # Parse discrete numeric columns
    print("\n解析离散数值列...")
    for col in TAX_DISCRETE_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"  {col}: 范围 [{df[col].min()}, {df[col].max()}]")

    data = df.values.astype(np.float32)
    output_path = os.path.join(data_dir, output_file)
    np.save(output_path, data)

    meta = {
        "all_cols": TAX_ALL_COLS,
        "categorical_cols": TAX_CATEGORICAL_COLS,
        "numeric_cols": TAX_NUMERIC_COLS,
        "continuous_cols": TAX_CONTINUOUS_COLS,
        "discrete_numeric_cols": TAX_DISCRETE_NUMERIC_COLS,
        "target_col": "",
        "num_features": int(data.shape[1]),
        "category_maps": label_maps,
        "note": (
            "Categorical columns (names, codes, location, marital status) are label-encoded. "
            "Salary and Rate are continuous. SingleExemp/MarriedExemp/ChildExemp are discrete numeric."
        ),
    }
    meta_path = os.path.join(data_dir, "tax_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n处理完成！")
    print(f"最终数据形状: {data.shape}")
    print(f"特征维度: {data.shape[1]}")
    print(f"样本数量: {data.shape[0]}")
    print(f"数据已保存至: {output_path}")

    print("\n列名与索引对应关系:")
    for i, col in enumerate(TAX_ALL_COLS):
        print(f"  {i}: {col}")

    return data, TAX_ALL_COLS


if __name__ == "__main__":
    preprocess_tax("tax.csv", "tax.npy")
