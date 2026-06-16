#!/usr/bin/env python
# coding: utf-8

"""BioCase Identification (r91800_c38) preprocessing for AR pipeline.

Dataset: t_biocase_identification_r91800_c38 (91799 rows, 38 columns)
Ground truth FDs (14):
  ScientificName -> SubspeciesEpithet, FullScientificNameString -> Subgenus,
  ScientificName -> SpeciesEpithet, FullScientificNameString -> SpeciesEpithet,
  FullScientificNameString -> GenusOrMonomial, FullScientificNameString -> SubspeciesEpithet,
  ScientificName -> FullScientificNameString, ScientificName -> GenusOrMonomial,
  ScientificName -> Subgenus, FullScientificNameString -> ScientificName,
  FullScientificNameString -> AuthorTeamParenthesisAndYear,
  FullScientificNameString -> AuthorTeamOriginalAndYear,
  ScientificName -> AuthorTeamOriginalAndYear,
  ScientificName -> AuthorTeamParenthesisAndYear

Columns kept (8):
  ScientificName                - 16835 unique → categorical
  FullScientificNameString      - 16835 unique → categorical
  GenusOrMonomial               - 3050 unique  → categorical
  Subgenus                      - 89 unique, 99.4% null → categorical
  SpeciesEpithet                - 6022 unique  → categorical
  SubspeciesEpithet             - 5405 unique, 36.9% null → categorical
  AuthorTeamOriginalAndYear     - 669 unique, 85.1% null → categorical
  AuthorTeamParenthesisAndYear  - 735 unique, 75.6% null → categorical

Columns dropped (30):
  _unitguid, _identificationguid, _timestamp - identifiers
  PreferredFlag - constant-like (2 values, not in GT)
  CombinationAuthorTeamAndYear through IdentificationHistory - 100% null or 99.7% null
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
    "ScientificName",
    "FullScientificNameString",
    "GenusOrMonomial",
    "Subgenus",
    "SpeciesEpithet",
    "SubspeciesEpithet",
    "AuthorTeamOriginalAndYear",
    "AuthorTeamParenthesisAndYear",
]

CATEGORICAL_COLS = [
    "ScientificName",
    "FullScientificNameString",
    "GenusOrMonomial",
    "Subgenus",
    "SpeciesEpithet",
    "SubspeciesEpithet",
    "AuthorTeamOriginalAndYear",
    "AuthorTeamParenthesisAndYear",
]

DISCRETE_NUMERIC_COLS: list[str] = []
NUMERIC_COLS: list[str] = []
CONTINUOUS_COLS: list[str] = []

DROPPED_COLUMNS = [
    "_unitguid",
    "_identificationguid",
    "_timestamp",
    "PreferredFlag",
    "CombinationAuthorTeamAndYear",
    "Breed",
    "NamedIndividual",
    "IdentificationQualifier",
    "IdentificationQualifier_insertionpoint",
    "NameAddendum",
    "InformalNameString",
    "InformalNameString_language",
    "Code",
    "NonFlag",
    "StoredUnderFlag",
    "ResultRole",
    "Date_DateText",
    "Date_TimeZone",
    "Date_ISODateTimeBegin",
    "Date_TimeOfDayBegin",
    "DayNumberBegin",
    "Date_ISODateTimeEnd",
    "Date_TimeOfDayEnd",
    "Date_DayNumberEnd",
    "PeriodExplicit",
    "Method",
    "Method_language",
    "Notes",
    "VerificationLevel",
    "IdentificationHistory",
]


def preprocess_biocase_identification(
    input_file: str = "t_biocase_identification_r91800_c38.csv",
    output_file: str = "biocase_identification.npy",
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
            "BioCase identification data (r91800, 8 effective columns). "
            "14 ground truth FDs among ScientificName, FullScientificNameString, "
            "GenusOrMonomial, Subgenus, SpeciesEpithet, SubspeciesEpithet, "
            "AuthorTeamOriginalAndYear, AuthorTeamParenthesisAndYear."
        ),
    }
    meta_path = os.path.join(data_dir, "biocase_identification_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Shape: {data.shape}, saved to {output_path}")
    return data, ALL_COLS


if __name__ == "__main__":
    preprocess_biocase_identification()
