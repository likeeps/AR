#!/usr/bin/env python
# coding: utf-8

"""Adult preprocessing for the modified FACE pipeline.

Key change from the original version:
- Only categorical columns are encoded here.
- Numerical columns are NOT standardized here.
- Standardization is done exactly once during training with train-split stats.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    from utils.dataUtils import (
        ADULT_ALL_COLS,
        ADULT_CATEGORICAL_COLS,
        ADULT_CONTINUOUS_COLS,
        ADULT_DISCRETE_NUMERIC_COLS,
        ADULT_NUMERIC_COLS,
        ADULT_TARGET_COL,
        resolve_project_path,
    )
except ImportError:
    from dataUtils import (
        ADULT_ALL_COLS,
        ADULT_CATEGORICAL_COLS,
        ADULT_CONTINUOUS_COLS,
        ADULT_DISCRETE_NUMERIC_COLS,
        ADULT_NUMERIC_COLS,
        ADULT_TARGET_COL,
        resolve_project_path,
    )


def preprocess_adult(
    input_file: str = "adult.csv",
    output_file: str = "adult.npy",
    project_path: str | None = None,
) -> Tuple[np.ndarray, list[str]]:
    # 所有输入输出统一走 project/traindata 目录，避免脚本执行目录不同导致路径错乱。
    project_path = resolve_project_path(project_path)
    data_dir = os.path.join(project_path, "traindata")
    os.makedirs(data_dir, exist_ok=True)

    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "outcome",
    ]

    input_file_path = os.path.join(data_dir, input_file)
    # 从 traindata/adult.csv 读取原始文件，列名固定，跳过原始文件自带表头。
    df = pd.read_csv(input_file_path, names=columns, skiprows=1, skipinitialspace=True, na_values="?")
    print(f"原始数据形状: {df.shape}")
    print(f"缺失值统计:\n{df.isnull().sum()}")

    df = df.dropna().reset_index(drop=True)
    print(f"删除缺失值后形状: {df.shape}")

    # 标签值统一映射到 0/1，供后续训练和查询流程复用。
    df[ADULT_TARGET_COL] = df[ADULT_TARGET_COL].str.strip().map({"<=50K": 0, ">50K": 1}).astype(np.int64)

    label_maps: Dict[str, Dict[str, int]] = {}

    # Only encode true categorical columns
    print("\n开始编码分类变量（仅真正的类别字段）...")
    for col in ADULT_CATEGORICAL_COLS:
        # 保留每个类别到整数 id 的映射，评估时可直接用类别字符串查询。
        unique_values = df[col].astype(str).unique()
        value_to_idx = {value: idx for idx, value in enumerate(unique_values)}
        df[col] = df[col].astype(str).map(value_to_idx)
        label_maps[col] = value_to_idx
        print(f"  {col}: {len(unique_values)} 个类别")

    # Keep numeric columns as-is (continuous and discrete numeric)
    print("\n保留数值变量的原始值...")
    for col in ADULT_CONTINUOUS_COLS + ADULT_DISCRETE_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"  {col}: 数值类型，范围 [{df[col].min():.2f}, {df[col].max():.2f}]")

    df = df[ADULT_ALL_COLS]
    # 训练脚本直接读取该 npy，不在预处理阶段做标准化。
    data = df.values.astype(np.float32)
    np.save(os.path.join(data_dir, output_file), data)

    meta = {
        "all_cols": ADULT_ALL_COLS,
        "numeric_cols": ADULT_NUMERIC_COLS,
        "numerical_cols": ADULT_NUMERIC_COLS,
        "continuous_cols": ADULT_CONTINUOUS_COLS,
        "discrete_numeric_cols": ADULT_DISCRETE_NUMERIC_COLS,
        "categorical_cols": ADULT_CATEGORICAL_COLS,
        "target_col": ADULT_TARGET_COL,
        "num_features": int(data.shape[1]),
        "category_maps": label_maps,
        "note": "Categorical columns are label-encoded. Numeric columns (continuous + discrete) keep raw values. No standardization here - training performs single-pass standardization.",
    }
    with open(os.path.join(data_dir, "adult_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n处理完成！")
    print(f"最终数据形状: {data.shape}")
    print(f"特征维度: {data.shape[1]}")
    print(f"样本数量: {data.shape[0]}")
    print(f"数据已保存至: {os.path.join(data_dir, output_file)}")

    print("\n列名与索引对应关系:")
    for i, col in enumerate(ADULT_ALL_COLS):
        print(f"  {i}: {col}")

    return data, ADULT_ALL_COLS


if __name__ == "__main__":
    preprocess_adult("adult.csv", "adult.npy")
