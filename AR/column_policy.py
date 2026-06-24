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


@dataclass(frozen=True)
class ColumnSemantics:
    name: str
    source_family: str
    role: str
    analysis_mode: str
    rhs_mode: str
    model_type: str
    search_space_mode: str
    is_target: bool
    constant_like: bool
    policy_lhs_capable: bool
    policy_rhs_capable: bool
    searchable_lhs: bool
    searchable_rhs: bool

    def to_searchability_profile(self) -> Dict[str, bool | str]:
        return {
            "search_space_mode": self.search_space_mode,
            "model_type": self.model_type,
            "is_target": bool(self.is_target),
            "constant_like": bool(self.constant_like),
            "policy_lhs_capable": bool(self.policy_lhs_capable),
            "policy_rhs_capable": bool(self.policy_rhs_capable),
            "searchable_lhs": bool(self.searchable_lhs),
            "searchable_rhs": bool(self.searchable_rhs),
            "runtime_searchable_lhs": bool(self.searchable_lhs),
            "runtime_searchable_rhs": bool(self.searchable_rhs),
            "lhs_source": self._source_label(self.policy_lhs_capable, self.searchable_lhs),
            "rhs_source": self._source_label(self.policy_rhs_capable, self.searchable_rhs),
            "lhs_runtime_blocked": bool(self.policy_lhs_capable and not self.searchable_lhs),
            "rhs_runtime_blocked": bool(self.policy_rhs_capable and not self.searchable_rhs),
        }

    @staticmethod
    def _source_label(policy_enabled: bool, runtime_enabled: bool) -> str:
        if not runtime_enabled:
            return "disabled"
        return "policy" if policy_enabled else "relaxed"


def normalize_search_space_mode(mode: object, default: str = "balanced") -> str:
    normalized_default = str(default or "balanced").strip().lower()
    if normalized_default not in {"strict", "balanced", "permissive"}:
        normalized_default = "balanced"

    normalized = str(mode or normalized_default).strip().lower()
    if normalized not in {"strict", "balanced", "permissive"}:
        return normalized_default
    return normalized


def is_constant_like_policy(policy) -> bool:
    return str(getattr(policy, "role", "")) == "constant" or int(getattr(policy, "unique_count", 0)) <= 1


def policy_searchable_flags(policy) -> tuple[bool, bool]:
    return policy.analysis_mode in {"exact", "binned"}, policy.rhs_mode == "exact"


def infer_policy_model_type(
    policy,
    *,
    is_categorical: bool,
    is_continuous: bool,
) -> str:
    if bool(is_categorical) or policy.family == "categorical":
        return "categorical"
    if bool(is_continuous) and policy.analysis_mode != "exact":
        return "continuous_bucket"
    if policy.analysis_mode == "binned":
        return "continuous_bucket"
    return "discrete_numeric"


def resolve_column_semantics(
    name: str,
    policy,
    *,
    is_categorical: bool,
    is_continuous: bool,
    is_target: bool,
    search_space_mode: object,
) -> ColumnSemantics:
    mode = normalize_search_space_mode(search_space_mode)
    model_type = infer_policy_model_type(
        policy,
        is_categorical=is_categorical,
        is_continuous=is_continuous,
    )
    strict_lhs, strict_rhs = policy_searchable_flags(policy)
    constant_like = is_constant_like_policy(policy)

    searchable_lhs = False
    searchable_rhs = False
    role = str(getattr(policy, "role", ""))

    if not bool(is_target) and not constant_like:
        if role == "near_identifier":
            if mode == "permissive":
                searchable_lhs, searchable_rhs = True, True
            elif mode == "balanced":
                searchable_lhs, searchable_rhs = True, True
        elif role in {"identifier", "quasi_identifier"}:
            if policy.unique_ratio >= 1.0:
                searchable_lhs, searchable_rhs = False, False
            else:
                searchable_lhs = True
                searchable_rhs = bool(strict_rhs and mode in {"balanced", "permissive"})
        elif mode == "strict":
            searchable_lhs, searchable_rhs = strict_lhs, strict_rhs
        else:
            searchable_lhs = model_type in {"categorical", "discrete_numeric", "continuous_bucket"}
            if mode == "permissive":
                searchable_rhs = model_type in {"categorical", "discrete_numeric", "continuous_bucket"}
            else:
                searchable_rhs = model_type in {"categorical", "discrete_numeric"}
            searchable_lhs = strict_lhs or searchable_lhs
            searchable_rhs = strict_rhs or searchable_rhs

    return ColumnSemantics(
        name=str(name),
        source_family=str(getattr(policy, "family", "")),
        role=role,
        analysis_mode=str(getattr(policy, "analysis_mode", "")),
        rhs_mode=str(getattr(policy, "rhs_mode", "")),
        model_type=model_type,
        search_space_mode=mode,
        is_target=bool(is_target),
        constant_like=bool(constant_like),
        policy_lhs_capable=bool(strict_lhs),
        policy_rhs_capable=bool(strict_rhs),
        searchable_lhs=bool(searchable_lhs),
        searchable_rhs=bool(searchable_rhs),
    )


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
                # quasi_identifier preserves exact matching semantics for ID-like
                # columns while still letting candidate validation control RHS risk.
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
    "ColumnSemantics",
    "build_column_policy_map",
    "infer_column_family",
    "infer_policy_model_type",
    "is_constant_like_policy",
    "normalize_search_space_mode",
    "policy_searchable_flags",
    "resolve_column_semantics",
]
