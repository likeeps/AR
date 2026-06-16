from __future__ import annotations

import numpy as np


IDENTIFIER_HINT_ROLES = frozenset(
    {
        "high_card_categorical",
        "entity_name",
        "geo_code",
        "code_categorical",
    }
)


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def ramp(value: float, start: float, end: float) -> float:
    if end <= start:
        return 1.0 if value >= end else 0.0
    return float(np.clip((value - start) / (end - start), 0.0, 1.0))


def inverse_ramp(value: float, start: float, end: float) -> float:
    return float(1.0 - ramp(value, start, end))


def column_identifier_signal(
    role: str,
    unique_ratio: float,
    empirical_weight: float,
    *,
    lhs_identifier_start_ratio: float,
    lhs_identifier_full_ratio: float,
) -> float:
    role_name = str(role or "")
    if role_name in {"identifier", "quasi_identifier"}:
        return 1.0

    unique_ratio_signal = ramp(float(unique_ratio), 0.02, 0.20)
    empirical_hint = clip01(float(empirical_weight))
    if role_name in IDENTIFIER_HINT_ROLES:
        return max(unique_ratio_signal, empirical_hint)
    return max(
        unique_ratio_signal,
        ramp(
            float(unique_ratio),
            lhs_identifier_start_ratio,
            lhs_identifier_full_ratio,
        ),
    )


def estimate_pattern_count(
    observed_unique_count: float,
    observed_unique_ratio: float,
    observed_rows: int,
    total_rows: int,
) -> float:
    observed = max(float(observed_unique_count), 0.0)
    if observed <= 0.0:
        return 0.0

    observed_row_count = max(int(observed_rows), 1)
    total_row_count = max(int(total_rows), observed_row_count)
    scaled = float(observed_unique_ratio) * float(total_row_count)
    scaled = max(scaled, observed * (float(total_row_count) / float(observed_row_count)))
    return float(min(max(observed, scaled), float(total_row_count)))
