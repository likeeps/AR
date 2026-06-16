#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ColumnPolicyConfig:
    exclude_unique_ratio: float = 0.98
    near_identifier_ratio: float = 0.50   # categorical columns above this are near-unique keys
    near_unique_numeric_ratio: float = 0.50
    exact_numeric_max_unique: int = 128
    exact_numeric_max_ratio: float = 0.05
    exact_date_max_unique: int = 32
    quasi_identifier_min_unique: int = 1024
    quasi_identifier_min_ratio: float = 0.03
    high_card_categorical_min_unique: int = 256
    high_card_categorical_min_ratio: float = 0.01
    quasi_identifier_empirical_weight: float = 0.95
    geo_code_empirical_weight: float = 0.70
    code_categorical_empirical_weight: float = 0.80
    entity_name_empirical_weight: float = 0.65
    location_categorical_empirical_weight: float = 0.55
    high_card_categorical_empirical_weight: float = 0.70


@dataclass(frozen=True)
class ColumnPolicy:
    name: str
    family: str
    role: str
    analysis_mode: str
    rhs_mode: str
    empirical_weight: float
    unique_count: int
    unique_ratio: float

    @property
    def score_source(self) -> str:
        if self.empirical_weight >= 1.0:
            return "empirical"
        if self.empirical_weight > 0.0:
            return "hybrid"
        return "model"

    def to_profile(self) -> Dict[str, float | int | str]:
        return {
            "family": self.family,
            "role": self.role,
            "mode": self.analysis_mode,
            "rhs_mode": self.rhs_mode,
            "score_source": self.score_source,
            "empirical_weight": float(self.empirical_weight),
            "unique_count": int(self.unique_count),
            "unique_ratio": float(self.unique_ratio),
        }


def infer_column_family(
    col_id: int,
    schema: List[str],
    discrete_ids: set[int],
    dataset_metadata: Optional[dict] = None,
    preprocessor=None,
) -> str:
    dataset_metadata = dataset_metadata or {}
    name = schema[col_id]
    categorical_names = set(dataset_metadata.get("categorical_cols", []))
    numeric_names = set(dataset_metadata.get("numeric_cols", []))
    date_names = set(dataset_metadata.get("date_cols", []))

    if preprocessor is not None:
        categorical_names.update(getattr(preprocessor.meta, "categorical_cols", []))
        numeric_names.update(getattr(preprocessor.meta, "numeric_cols", []))
        if name in getattr(preprocessor.meta, "category_maps", {}):
            categorical_names.add(name)

    if name in date_names:
        return "date"
    if name in categorical_names:
        return "categorical"
    if col_id in discrete_ids:
        return "discrete_numeric"
    if name in numeric_names:
        return "numeric"
    return "unknown"


def build_column_policy_map(
    schema: List[str],
    discrete_ids: set[int],
    dataset_metadata: Optional[dict] = None,
    preprocessor=None,
    column_stats: Optional[Dict[int, Dict[str, float]]] = None,
    config: Optional[ColumnPolicyConfig] = None,
) -> Dict[int, ColumnPolicy]:
    dataset_metadata = dataset_metadata or {}
    column_stats = column_stats or {}
    config = config or ColumnPolicyConfig()

    policies: Dict[int, ColumnPolicy] = {}

    for col_id, name in enumerate(schema):
        name_lower = name.strip().lower()
        stats = column_stats.get(int(col_id), {})
        unique_count = int(stats.get("unique_count", 0))
        unique_ratio = float(stats.get("unique_ratio", 0.0))
        family = infer_column_family(
            col_id=col_id,
            schema=schema,
            discrete_ids=discrete_ids,
            dataset_metadata=dataset_metadata,
            preprocessor=preprocessor,
        )

        role = "excluded"
        analysis_mode = "excluded"
        rhs_mode = "excluded"
        empirical_weight = 0.0

        is_high_card = (
            unique_count >= config.high_card_categorical_min_unique
            or unique_ratio >= config.high_card_categorical_min_ratio
        )
        is_quasi_card = (
            unique_count >= config.quasi_identifier_min_unique
            and unique_ratio >= config.quasi_identifier_min_ratio
        )
        has_strong_id_signal = any(
            token in name_lower for token in ("id", "uuid", "guid", "number", "phone", "email", "account")
        )
        has_geo_code_signal = any(token in name_lower for token in ("zip", "postal"))
        has_code_signal = "avg" in name_lower or (
            "code" in name_lower and not has_geo_code_signal
        )
        has_entity_name_signal = "name" in name_lower
        has_location_signal = any(
            token in name_lower for token in ("city", "county", "state", "country")
        )

        if unique_count <= 1:
            role = "constant"
        elif unique_ratio >= config.exclude_unique_ratio:
            role = "identifier"
            empirical_weight = 1.0
        elif family == "categorical":
            if unique_ratio >= config.near_identifier_ratio and has_strong_id_signal:
                # Columns with identifier-like names (guid, id, uuid, etc.) and high
                # unique_ratio should be quasi_identifier, not near_identifier.
                # near_identifier blocks the column entirely; quasi_identifier allows
                # LHS usage (important for FDs like _unitguid->_datasetguid).
                role = "quasi_identifier"
                analysis_mode = "exact"
                rhs_mode = "exact"
                empirical_weight = config.quasi_identifier_empirical_weight
            elif unique_ratio >= config.near_identifier_ratio:
                role = "near_identifier"
            elif is_high_card and has_strong_id_signal and is_quasi_card:
                role = "quasi_identifier"
                analysis_mode = "exact"
                rhs_mode = "exact"
                empirical_weight = config.quasi_identifier_empirical_weight
            elif is_high_card and has_geo_code_signal:
                role = "geo_code"
                analysis_mode = "exact"
                rhs_mode = "exact"
                empirical_weight = config.geo_code_empirical_weight
            elif is_high_card and has_code_signal:
                role = "code_categorical"
                analysis_mode = "exact"
                rhs_mode = "exact"
                empirical_weight = config.code_categorical_empirical_weight
            elif is_high_card and has_location_signal:
                role = "location_categorical"
                analysis_mode = "exact"
                rhs_mode = "exact"
                empirical_weight = config.location_categorical_empirical_weight
            elif is_high_card and has_entity_name_signal:
                role = "entity_name"
                analysis_mode = "exact"
                rhs_mode = "exact"
                empirical_weight = config.entity_name_empirical_weight
            elif is_high_card:
                role = "high_card_categorical"
                analysis_mode = "exact"
                rhs_mode = "exact"
                empirical_weight = config.high_card_categorical_empirical_weight
            else:
                role = "semantic_categorical"
                analysis_mode = "exact"
                rhs_mode = "exact"
        elif family == "date":
            role = "temporal"
            if unique_ratio >= config.near_unique_numeric_ratio:
                analysis_mode = "excluded"
            elif (
                unique_count <= config.exact_date_max_unique
                and unique_ratio <= config.exact_numeric_max_ratio
            ):
                analysis_mode = "exact"
                rhs_mode = "exact"
            else:
                analysis_mode = "binned"
        elif family in {"numeric", "discrete_numeric", "unknown"}:
            role = "discrete_numeric" if family == "discrete_numeric" else "continuous_numeric"
            if unique_ratio >= config.near_unique_numeric_ratio:
                analysis_mode = "excluded"
            elif (
                unique_count <= config.exact_numeric_max_unique
                and unique_ratio <= config.exact_numeric_max_ratio
            ):
                analysis_mode = "exact"
                rhs_mode = "exact"
            else:
                analysis_mode = "binned"

        policies[col_id] = ColumnPolicy(
            name=name,
            family=family,
            role=role,
            analysis_mode=analysis_mode,
            rhs_mode=rhs_mode,
            empirical_weight=float(empirical_weight),
            unique_count=unique_count,
            unique_ratio=unique_ratio,
        )

    return policies


__all__ = [
    "ColumnPolicy",
    "ColumnPolicyConfig",
    "build_column_policy_map",
    "infer_column_family",
]
