#!/usr/bin/env python
# coding: utf-8

"""Claims preprocessing for normalizing flow training.

Dataset: TSA Claims (97k+ records, 13 columns)
Key characteristics:
- ClaimNumber: unique identifier (should be excluded)
- Dates: DateReceived, IncidentDate (convert to numeric)
- Categorical: AirportCode, AirportName, AirlineName, ClaimType, ClaimSite, Item, Status, Disposition
- Numeric: ClaimAmount, CloseAmount (need cleaning)
- High missing rate in some columns
"""

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


# Schema definition
CLAIMS_ALL_COLS = [
    "DateReceived", "IncidentDate", "AirportCode", "AirportName", "AirlineName",
    "ClaimType", "ClaimSite", "Item", "ClaimAmount", "Status", "CloseAmount", "Disposition"
]

# Exclude ClaimNumber (unique identifier, acts as primary key)
CLAIMS_CATEGORICAL_COLS = [
    "AirportCode", "AirportName", "AirlineName", "ClaimType", "ClaimSite",
    "Item", "Status", "Disposition"
]

CLAIMS_NUMERIC_COLS = [
    "ClaimAmount", "CloseAmount"
]

CLAIMS_DATE_COLS = [
    "DateReceived", "IncidentDate"
]


def parse_date_to_days(date_str):
    """Convert date string to days since epoch (simple numeric representation)."""
    if pd.isna(date_str) or date_str == '':
        return np.nan
    try:
        dt = pd.to_datetime(date_str, errors='coerce')
        if pd.isna(dt):
            return np.nan
        # Days since 2000-01-01
        epoch = pd.Timestamp('2000-01-01')
        return (dt - epoch).days
    except:
        return np.nan


def parse_currency(value):
    """Parse currency string like '$350,00' to float."""
    if pd.isna(value) or value == '':
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    # Remove $, commas, spaces
    cleaned = re.sub(r'[\$,\s]', '', str(value))
    try:
        return float(cleaned)
    except:
        return np.nan


def preprocess_claims(
    input_file: str = "claims.csv",
    output_file: str = "claims.npy",
    project_path: str | None = None,
) -> Tuple[np.ndarray, list[str]]:
    project_path = resolve_project_path(project_path)
    data_dir = os.path.join(project_path, "traindata")
    os.makedirs(data_dir, exist_ok=True)

    input_file_path = os.path.join(data_dir, input_file)
    df = pd.read_csv(input_file_path, low_memory=False)
    print(f"原始数据形状: {df.shape}")
    print(f"缺失值统计:\n{df.isnull().sum()}")

    # Drop ClaimNumber (unique identifier)
    if 'ClaimNumber' in df.columns:
        df = df.drop(columns=['ClaimNumber'])
        print("已删除 ClaimNumber 列（唯一标识符）")

    # Parse dates to numeric
    for col in CLAIMS_DATE_COLS:
        if col in df.columns:
            df[col] = df[col].apply(parse_date_to_days)
            print(f"日期列 {col} 已转换为数值（距2000-01-01的天数）")

    # Parse currency columns
    for col in CLAIMS_NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].apply(parse_currency)
            print(f"货币列 {col} 已解析为数值")

    # Drop rows with too many missing values (>50% columns missing)
    threshold = len(df.columns) * 0.5
    df = df.dropna(thresh=threshold).reset_index(drop=True)
    print(f"删除缺失值过多的行后形状: {df.shape}")

    # Reorder columns
    df = df[CLAIMS_ALL_COLS]

    # Encode categorical columns
    label_maps: Dict[str, Dict[str, int]] = {}
    print("\n开始编码分类变量...")
    for col in CLAIMS_CATEGORICAL_COLS:
        if col in df.columns:
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

    # Keep NaN for numeric/date columns (AR pipeline handles NaN → [NULL])
    for col in CLAIMS_NUMERIC_COLS + CLAIMS_DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    data = df.values.astype(np.float32)
    np.save(os.path.join(data_dir, output_file), data)

    meta = {
        "all_cols": CLAIMS_ALL_COLS,
        "categorical_cols": CLAIMS_CATEGORICAL_COLS,
        "numeric_cols": CLAIMS_NUMERIC_COLS,
        "continuous_cols": CLAIMS_NUMERIC_COLS,
        "discrete_numeric_cols": [],
        "date_cols": CLAIMS_DATE_COLS,
        "target_col": "",
        "num_features": int(data.shape[1]),
        "category_maps": label_maps,
        "note": "ClaimNumber excluded. Dates converted to days since 2000-01-01. Currency parsed. Missing values filled.",
    }
    with open(os.path.join(data_dir, "claims_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n处理完成！")
    print(f"最终数据形状: {data.shape}")
    print(f"特征维度: {data.shape[1]}")
    print(f"样本数量: {data.shape[0]}")
    print(f"数据已保存至: {os.path.join(data_dir, output_file)}")

    print("\n列名与索引对应关系:")
    for i, col in enumerate(CLAIMS_ALL_COLS):
        print(f"  {i}: {col}")

    return data, CLAIMS_ALL_COLS


if __name__ == "__main__":
    preprocess_claims("claims.csv", "claims.npy")
