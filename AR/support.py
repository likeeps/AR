from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from AR.config import SearchConfig
from AR.schema import DatasetSchema


@dataclass
class SupportTable:
    lhs_cols: tuple[int, ...]
    lhs_values: np.ndarray
    counts: np.ndarray
    effective_counts: np.ndarray
    probabilities: np.ndarray
    empirical_weights: np.ndarray
    retained_mass: float
    effective_retained_mass: float
    total_patterns: int
    raw_probabilities: np.ndarray | None = None
    dropped_mass: float = 0.0
    non_null_ratios: np.ndarray | None = None
    empirical_entropies: np.ndarray | None = None
    empirical_top1: np.ndarray | None = None

    @property
    def num_rows(self) -> int:
        return int(self.lhs_values.shape[0])

    def iter_evidences(self) -> list[dict[int, int]]:
        if self.num_rows == 0:
            return []
        if self.lhs_values.ndim == 1:
            return [{self.lhs_cols[0]: int(value)} for value in self.lhs_values.tolist()]
        return [
            {col_id: int(value) for col_id, value in zip(self.lhs_cols, row.tolist())}
            for row in self.lhs_values
        ]


class SupportEstimator:
    def __init__(self, train_tokens: np.ndarray, schema: DatasetSchema, config: SearchConfig) -> None:
        self.train_tokens = np.asarray(train_tokens, dtype=np.int64)
        self.schema = schema
        self.config = config
        self.n_rows = int(self.train_tokens.shape[0])
        self._special_ids = {
            col_id: {column.null_id, column.unk_id, column.mask_id, column.rare_id}
            for col_id, column in enumerate(schema.columns)
        }
        self._marginals = [
            np.bincount(self.train_tokens[:, col_id], minlength=column.vocab_size).astype(np.float64) / max(self.n_rows, 1)
            for col_id, column in enumerate(schema.columns)
        ]

    def marginal_distribution(self, rhs_col: int) -> np.ndarray:
        return self._marginals[int(rhs_col)].copy()

    def valid_rhs_mask(self, rhs_col: int) -> np.ndarray:
        """Boolean mask over vocab: True for non-special tokens."""
        col = self.schema.columns[int(rhs_col)]
        mask = np.ones(col.vocab_size, dtype=bool)
        for special_id in self._special_ids[int(rhs_col)]:
            mask[int(special_id)] = False
        return mask

    def marginal_distribution_valid(self, rhs_col: int) -> np.ndarray:
        """Marginal distribution over non-special tokens, renormalized to sum to 1."""
        raw = self._marginals[int(rhs_col)].copy()
        mask = self.valid_rhs_mask(rhs_col)
        raw[~mask] = 0.0
        total = raw.sum()
        if total > 0:
            raw /= total
        return raw

    def _build_single_support(self, lhs_col: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values, inverse, counts = np.unique(self.train_tokens[:, lhs_col], return_inverse=True, return_counts=True)
        return values.astype(np.int64), counts.astype(np.int64), inverse.astype(np.int64)

    def _build_pair_support(self, lhs_cols: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        left, right = lhs_cols
        base = int(self.schema.columns[right].vocab_size)
        keys = self.train_tokens[:, left].astype(np.int64) * base + self.train_tokens[:, right].astype(np.int64)
        unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        left_values = (unique_keys // base).astype(np.int64)
        right_values = (unique_keys % base).astype(np.int64)
        values = np.stack([left_values, right_values], axis=1)
        return values, counts.astype(np.int64), inverse.astype(np.int64)

    def _build_multi_support(self, lhs_cols: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Encode each row as a tuple-string key for arbitrary LHS size
        col_data = [self.train_tokens[:, col_id].astype(np.int64) for col_id in lhs_cols]
        # Use structured array approach: encode as mixed-radix integer if vocab sizes allow,
        # otherwise fall back to string keys via np.unique on structured array
        # Use void-view trick for fast unique on multi-column
        combined = np.stack(col_data, axis=1)
        # View as void bytes for np.unique
        dt = np.dtype((np.void, combined.dtype.itemsize * combined.shape[1]))
        combined_view = np.ascontiguousarray(combined).view(dt).ravel()
        unique_keys, inverse, counts = np.unique(combined_view, return_inverse=True, return_counts=True)
        # Recover original values
        values = unique_keys.view(combined.dtype).reshape(-1, len(lhs_cols))
        return values.astype(np.int64), counts.astype(np.int64), inverse.astype(np.int64)

    def _special_mask(self, lhs_cols: tuple[int, ...], lhs_values: np.ndarray) -> np.ndarray:
        if lhs_values.ndim == 1:
            return ~np.isin(lhs_values, list(self._special_ids[lhs_cols[0]]))

        keep_mask = np.ones(lhs_values.shape[0], dtype=bool)
        for axis, col_id in enumerate(lhs_cols):
            keep_mask &= ~np.isin(lhs_values[:, axis], list(self._special_ids[col_id]))
        return keep_mask

    def _empirical_rhs_stats(
        self,
        rhs_col: int,
        counts: np.ndarray,
        inverse: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rhs_values = self.train_tokens[:, rhs_col].astype(np.int64)
        valid_mask = ~np.isin(rhs_values, list(self._special_ids[int(rhs_col)]))
        valid_group_ids = inverse[valid_mask].astype(np.int64)
        valid_rhs_values = rhs_values[valid_mask]
        valid_counts = np.zeros(counts.shape[0], dtype=np.int64)
        if valid_group_ids.size > 0:
            np.add.at(valid_counts, valid_group_ids, 1)

        entropies = np.zeros(counts.shape[0], dtype=np.float64)
        top1 = np.zeros(counts.shape[0], dtype=np.float64)
        if valid_group_ids.size == 0:
            non_null_ratios = valid_counts.astype(np.float64) / np.maximum(counts.astype(np.float64), 1.0)
            return entropies, top1, valid_counts, non_null_ratios

        rhs_vocab = int(self.schema.columns[rhs_col].vocab_size)
        pair_keys = valid_group_ids.astype(np.int64) * rhs_vocab + valid_rhs_values
        unique_pairs, pair_counts = np.unique(pair_keys, return_counts=True)

        group_ids = (unique_pairs // rhs_vocab).astype(np.int64)
        top1_counts = np.zeros(counts.shape[0], dtype=np.int64)
        np.maximum.at(top1_counts, group_ids, pair_counts.astype(np.int64))

        probs = pair_counts.astype(np.float64) / np.maximum(valid_counts[group_ids].astype(np.float64), 1.0)
        entropy_terms = -probs * np.log(np.clip(probs, 1e-12, 1.0))
        np.add.at(entropies, group_ids, entropy_terms)

        valid_mask_groups = valid_counts > 0
        top1[valid_mask_groups] = (
            top1_counts[valid_mask_groups].astype(np.float64)
            / np.maximum(valid_counts[valid_mask_groups].astype(np.float64), 1.0)
        )
        non_null_ratios = valid_counts.astype(np.float64) / np.maximum(counts.astype(np.float64), 1.0)
        return entropies, top1, valid_counts, non_null_ratios

    def _select_support_indices(self, priority: np.ndarray) -> np.ndarray:
        num_rows = int(priority.shape[0])
        max_rows = int(self.config.max_support_rows)
        if num_rows <= max_rows:
            return np.arange(num_rows, dtype=np.int64)

        head_rows = int(min(max(self.config.support_head_rows, 0), max_rows))
        head_indices = np.arange(head_rows, dtype=np.int64)
        remaining_budget = max_rows - head_rows
        if remaining_budget <= 0:
            return head_indices

        tail_start = head_rows
        tail_count = num_rows - tail_start
        if tail_count <= remaining_budget:
            return np.arange(num_rows, dtype=np.int64)

        tail_priority = priority[tail_start:].astype(np.float64)
        if float(tail_priority.sum()) <= 0.0:
            relative = np.linspace(0, tail_count - 1, num=remaining_budget, dtype=np.int64)
            tail_indices = tail_start + relative
        else:
            quantiles = (np.arange(remaining_budget, dtype=np.float64) + 0.5) / remaining_budget
            cdf = np.cumsum(tail_priority) / tail_priority.sum()
            relative = np.searchsorted(cdf, quantiles, side="left").astype(np.int64)
            tail_indices = tail_start + np.clip(relative, 0, tail_count - 1)
            tail_indices = np.unique(tail_indices)

            if tail_indices.shape[0] < remaining_budget:
                fill_budget = remaining_budget - tail_indices.shape[0]
                mask = np.ones(tail_count, dtype=bool)
                mask[tail_indices - tail_start] = False
                available = np.nonzero(mask)[0]
                if available.size > 0:
                    filler = available[
                        np.linspace(0, available.size - 1, num=min(fill_budget, available.size), dtype=np.int64)
                    ]
                    tail_indices = np.sort(np.concatenate([tail_indices, tail_start + filler]))

        selected = np.concatenate([head_indices, tail_indices[:remaining_budget]])
        return np.asarray(selected, dtype=np.int64)

    def _support_priority(
        self,
        counts: np.ndarray,
        effective_counts: np.ndarray | None,
        non_null_ratios: np.ndarray | None,
    ) -> np.ndarray:
        base_counts = counts.astype(np.float64)
        if effective_counts is not None:
            base_counts = np.maximum(effective_counts.astype(np.float64), 0.0)

        alpha = float(max(self.config.support_priority_effective_alpha, 0.0))
        beta = float(max(self.config.support_priority_non_null_beta, 0.0))
        priority = np.power(base_counts + 1.0, alpha)
        if non_null_ratios is not None:
            priority *= np.power(np.clip(non_null_ratios.astype(np.float64), 1e-6, 1.0), beta)
        return np.asarray(priority, dtype=np.float64)

    def _column_unique_ratio(self, col_id: int) -> float:
        column = self.schema.columns[int(col_id)]
        policy = dict(column.stats.get("policy") or {})
        unique_ratio = float(policy.get("unique_ratio", 0.0) or 0.0)
        if unique_ratio <= 0.0:
            train_unique = float(column.stats.get("train_unique", 0.0) or 0.0)
            unique_ratio = train_unique / max(float(self.n_rows), 1.0)
        return float(unique_ratio)

    def _has_high_card_lhs(self, lhs_cols: tuple[int, ...]) -> bool:
        threshold = float(max(self.config.high_card_lhs_unique_ratio, 0.0))
        return any(self._column_unique_ratio(col_id) > threshold for col_id in lhs_cols)

    @lru_cache(maxsize=256)
    def build_support(self, lhs_cols: tuple[int, ...], rhs_col: int | None = None) -> SupportTable:
        lhs_cols = tuple(sorted(lhs_cols))
        if len(lhs_cols) == 1:
            lhs_values, counts, inverse = self._build_single_support(lhs_cols[0])
        elif len(lhs_cols) == 2:
            lhs_values, counts, inverse = self._build_pair_support((lhs_cols[0], lhs_cols[1]))
        else:
            lhs_values, counts, inverse = self._build_multi_support(lhs_cols)

        total_patterns = int(counts.shape[0])
        all_counts = counts
        all_empirical_entropies: np.ndarray | None = None
        all_empirical_top1: np.ndarray | None = None
        all_effective_counts: np.ndarray | None = None
        all_non_null_ratios: np.ndarray | None = None
        if rhs_col is not None:
            (
                all_empirical_entropies,
                all_empirical_top1,
                all_effective_counts,
                all_non_null_ratios,
            ) = self._empirical_rhs_stats(
                int(rhs_col),
                counts=np.asarray(all_counts, dtype=np.int64),
                inverse=inverse,
            )

        special_mask = self._special_mask(lhs_cols, lhs_values)
        lhs_values = lhs_values[special_mask]
        counts = counts[special_mask]
        empirical_entropies: np.ndarray | None = None
        empirical_top1: np.ndarray | None = None
        effective_counts: np.ndarray | None = None
        non_null_ratios: np.ndarray | None = None
        if all_empirical_entropies is not None and all_empirical_top1 is not None:
            empirical_entropies = all_empirical_entropies[special_mask]
            empirical_top1 = all_empirical_top1[special_mask]
        if all_effective_counts is not None and all_non_null_ratios is not None:
            effective_counts = all_effective_counts[special_mask]
            non_null_ratios = all_non_null_ratios[special_mask]

        has_high_card_lhs = self._has_high_card_lhs(lhs_cols)
        effective_support_floor = int(self.config.min_effective_support_count)
        pure_support_floor = int(self.config.min_pure_support_count)
        if has_high_card_lhs:
            high_card_floor = int(self.config.high_card_min_effective_support_count)
            effective_support_floor = max(effective_support_floor, high_card_floor)
            pure_support_floor = max(pure_support_floor, high_card_floor)

        support_mask = counts >= self.config.min_support_count
        if effective_counts is not None:
            support_mask |= effective_counts >= effective_support_floor
        if empirical_top1 is not None:
            if effective_counts is not None:
                support_mask |= (
                    (empirical_top1 >= self.config.min_empirical_row_purity)
                    & (effective_counts >= pure_support_floor)
                )
            else:
                support_mask |= empirical_top1 >= self.config.min_empirical_row_purity
        if non_null_ratios is not None and self.config.min_non_null_ratio > 0.0:
            support_mask &= non_null_ratios >= self.config.min_non_null_ratio
        lhs_values = lhs_values[support_mask]
        counts = counts[support_mask]
        if empirical_entropies is not None:
            empirical_entropies = empirical_entropies[support_mask]
        if empirical_top1 is not None:
            empirical_top1 = empirical_top1[support_mask]
        if effective_counts is not None:
            effective_counts = effective_counts[support_mask]
        if non_null_ratios is not None:
            non_null_ratios = non_null_ratios[support_mask]

        if lhs_values.size == 0:
            return SupportTable(
                lhs_cols=lhs_cols,
                lhs_values=np.zeros((0, len(lhs_cols)), dtype=np.int64),
                counts=np.zeros((0,), dtype=np.int64),
                effective_counts=np.zeros((0,), dtype=np.int64),
                probabilities=np.zeros((0,), dtype=np.float64),
                empirical_weights=np.zeros((0,), dtype=np.float64),
                retained_mass=0.0,
                effective_retained_mass=0.0,
                total_patterns=total_patterns,
                non_null_ratios=np.zeros((0,), dtype=np.float64),
                empirical_entropies=np.zeros((0,), dtype=np.float64),
                empirical_top1=np.zeros((0,), dtype=np.float64),
                raw_probabilities=np.zeros((0,), dtype=np.float64),
                dropped_mass=1.0,
            )

        priority = self._support_priority(counts, effective_counts, non_null_ratios)
        order = np.argsort(-priority)
        lhs_values = lhs_values[order]
        counts = counts[order]
        priority = priority[order]
        if empirical_entropies is not None:
            empirical_entropies = empirical_entropies[order]
        if empirical_top1 is not None:
            empirical_top1 = empirical_top1[order]
        if effective_counts is not None:
            effective_counts = effective_counts[order]
        if non_null_ratios is not None:
            non_null_ratios = non_null_ratios[order]

        selected_indices = self._select_support_indices(priority)
        lhs_values = lhs_values[selected_indices]
        counts = counts[selected_indices]
        if empirical_entropies is not None:
            empirical_entropies = empirical_entropies[selected_indices]
        if empirical_top1 is not None:
            empirical_top1 = empirical_top1[selected_indices]
        if effective_counts is not None:
            effective_counts = effective_counts[selected_indices]
        if non_null_ratios is not None:
            non_null_ratios = non_null_ratios[selected_indices]

        # Detect degenerate LHS patterns: a single value dominating the distribution
        # (e.g. "Unknown" from fillna on a 99.7%-null column).  These patterns produce
        # trivial empirical distributions (essentially the marginal) and drown out
        # the signal from the sparse non-null patterns that carry real FD information.
        # Soft penalty: linearly scale counts from 80% (no penalty) to 100% (full zero).
        degenerate_rows_removed = 0
        total_count = int(counts.sum())
        if total_count > 0:
            pattern_ratios = counts.astype(np.float64) / max(total_count, 1)
            degenerate_penalty = np.clip((pattern_ratios - 0.80) / 0.20, 0.0, 1.0)
            has_penalty = degenerate_penalty > 0.0
            if has_penalty.any() and not degenerate_penalty[degenerate_penalty > 0.0].all():
                # Compute effective removed mass for retained_mass denominator
                degenerate_rows_removed = int(float(np.sum(counts.astype(np.float64) * degenerate_penalty)))
                counts = counts.copy()
                counts = np.round(counts.astype(np.float64) * (1.0 - degenerate_penalty)).astype(np.int64)
                if empirical_entropies is not None:
                    empirical_entropies = empirical_entropies.copy()
                    empirical_entropies *= (1.0 - degenerate_penalty)
                if empirical_top1 is not None:
                    empirical_top1 = empirical_top1.copy()
                    empirical_top1 *= (1.0 - degenerate_penalty)
                if effective_counts is not None:
                    effective_counts = effective_counts.copy()
                    effective_counts = np.round(effective_counts.astype(np.float64) * (1.0 - degenerate_penalty)).astype(np.int64)
                if non_null_ratios is not None:
                    non_null_ratios = non_null_ratios.copy()
                    non_null_ratios *= (1.0 - degenerate_penalty)

        empirical = counts.astype(np.float64) / max(self.n_rows, 1)
        product = np.ones_like(empirical)
        empirical_weights = counts.astype(np.float64) / (counts.astype(np.float64) + self.config.support_beta)

        if lhs_values.ndim == 1:
            product *= self._marginals[lhs_cols[0]][lhs_values]
            lhs_values_export = lhs_values
        else:
            lhs_values_export = lhs_values
            for axis, col_id in enumerate(lhs_cols):
                product *= self._marginals[col_id][lhs_values[:, axis]]

        blend = counts.astype(np.float64) / (counts.astype(np.float64) + self.config.support_beta)
        independence_weight = float(np.clip(self.config.support_independence_weight, 0.0, 1.0))
        probabilities = empirical + independence_weight * (1.0 - blend) * product
        # Compute retained_mass: when degenerate patterns were filtered, use the
        # non-degenerate rows as denominator to avoid artificially low mass.
        mass_denominator = max(self.n_rows - degenerate_rows_removed, 1)
        retained_mass = float(counts.sum() / mass_denominator)
        effective_retained_mass = 0.0
        if effective_counts is not None:
            effective_retained_mass = float(effective_counts.sum() / mass_denominator)
        # raw_probabilities: unnormalized over kept patterns, sum ≈ retained_mass
        raw_probabilities = probabilities.copy()
        dropped_mass = max(1.0 - retained_mass, 0.0)
        probabilities = probabilities / max(probabilities.sum(), 1e-12)

        return SupportTable(
            lhs_cols=lhs_cols,
            lhs_values=np.asarray(lhs_values_export, dtype=np.int64),
            counts=np.asarray(counts, dtype=np.int64),
            effective_counts=(
                np.zeros((counts.shape[0],), dtype=np.int64)
                if effective_counts is None
                else np.asarray(effective_counts, dtype=np.int64)
            ),
            probabilities=np.asarray(probabilities, dtype=np.float64),
            empirical_weights=np.asarray(empirical_weights, dtype=np.float64),
            retained_mass=retained_mass,
            effective_retained_mass=effective_retained_mass,
            total_patterns=total_patterns,
            non_null_ratios=(
                None if non_null_ratios is None else np.asarray(non_null_ratios, dtype=np.float64)
            ),
            empirical_entropies=None if empirical_entropies is None else np.asarray(empirical_entropies, dtype=np.float64),
            empirical_top1=None if empirical_top1 is None else np.asarray(empirical_top1, dtype=np.float64),
            raw_probabilities=np.asarray(raw_probabilities, dtype=np.float64),
            dropped_mass=dropped_mass,
        )

    def support_probability(self, lhs_cols: tuple[int, ...], lhs_values: tuple[int, ...]) -> float:
        lhs_cols = tuple(sorted(lhs_cols))
        table = self.build_support(lhs_cols)
        if table.num_rows == 0:
            return 0.0

        if len(lhs_cols) == 1:
            lhs_value = int(lhs_values[0])
            matches = np.where(table.lhs_values == lhs_value)[0]
        else:
            lhs_vector = np.asarray(lhs_values, dtype=np.int64)
            matches = np.where(np.all(table.lhs_values == lhs_vector, axis=1))[0]

        if matches.size == 0:
            return 0.0
        return float(table.probabilities[int(matches[0])])
