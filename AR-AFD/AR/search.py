from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from AR.config import DEFAULT_DATASET_NAME, PipelineConfig, SUPPORTED_DATASETS, default_config, validate_config
from AR.metrics import entropy_from_probs
from AR.query_engine import QueryEngine
from AR.support import SupportEstimator, SupportTable


def parse_fd_file(file_path: str) -> set[tuple[tuple[str, ...], str]]:
    fd_collection: set[tuple[tuple[str, ...], str]] = set()
    with open(file_path, "r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            if "->" not in cleaned:
                raise ValueError(f"Invalid FD format in line {line_num}: {cleaned}")
            lhs_str, rhs_str = cleaned.split("->", 1)
            lhs = tuple(attr.strip().replace("-", "_") for attr in lhs_str.split(",") if attr.strip())
            rhs = rhs_str.strip().replace("-", "_")
            fd_collection.add((lhs, rhs))
    return fd_collection


def evaluate_fd(
    discovered_fds: list[tuple[list[str], str]],
    ground_truth_path: str,
) -> tuple[float, float, float]:
    truth = parse_fd_file(ground_truth_path)
    standardized_truth = {(tuple(sorted(lhs)), rhs) for lhs, rhs in truth}
    standardized_disc = {(tuple(sorted(lhs)), rhs) for lhs, rhs in discovered_fds}

    tp = len(standardized_disc & standardized_truth)
    fp = len(standardized_disc - standardized_truth)
    fn = len(standardized_truth - standardized_disc)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def write_discovered_report(
    discovered_fds: list[tuple[list[str], str]],
    output_path: str,
    precision: float,
    recall: float,
    f1: float,
    elapsed_ms: int,
) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    sorted_fds = sorted(discovered_fds, key=lambda item: (tuple(sorted(item[0])), item[1]))
    with open(output_path, "w", encoding="utf-8") as handle:
        for lhs, rhs in sorted_fds:
            handle.write(f"{','.join(sorted(lhs))}->{rhs}\n")
        handle.write(f"Precision: {precision:.4f}\n")
        handle.write(f"Recall: {recall:.4f}\n")
        handle.write(f"F1: {f1:.4f}\n")
        handle.write(f"Runtime: {elapsed_ms} ms\n")


@dataclass
class CandidateScore:
    rhs_col: int
    lhs_cols: tuple[int, ...]
    s_ent: float
    s_acc: float
    score: float
    model_score: float
    empirical_score: float
    empirical_blend: float
    empirical_bonus_weight: float
    coverage_factor: float
    group_factor: float
    model_s_ent: float
    model_s_acc: float
    empirical_s_ent: float
    empirical_s_acc: float
    reverse_score: float
    direction_margin: float
    support_rows: int
    effective_support_rows: int
    retained_mass: float
    effective_retained_mass: float
    weighted_non_null_ratio: float
    expected_entropy: float
    expected_top1: float
    empirical_expected_entropy: float
    empirical_expected_top1: float
    marginal_entropy: float
    marginal_top1: float
    bidirectional: bool = False

    def to_dict(self, engine: QueryEngine) -> dict[str, object]:
        return {
            "rhs": engine.schema.columns[self.rhs_col].name,
            "lhs": [engine.schema.columns[col_id].name for col_id in self.lhs_cols],
            "s_ent": self.s_ent,
            "s_acc": self.s_acc,
            "score": self.score,
            "model_score": self.model_score,
            "empirical_score": self.empirical_score,
            "empirical_blend": self.empirical_blend,
            "empirical_bonus_weight": self.empirical_bonus_weight,
            "coverage_factor": self.coverage_factor,
            "group_factor": self.group_factor,
            "model_s_ent": self.model_s_ent,
            "model_s_acc": self.model_s_acc,
            "empirical_s_ent": self.empirical_s_ent,
            "empirical_s_acc": self.empirical_s_acc,
            "reverse_score": self.reverse_score,
            "direction_margin": self.direction_margin,
            "support_rows": self.support_rows,
            "effective_support_rows": self.effective_support_rows,
            "retained_mass": self.retained_mass,
            "effective_retained_mass": self.effective_retained_mass,
            "weighted_non_null_ratio": self.weighted_non_null_ratio,
            "expected_entropy": self.expected_entropy,
            "expected_top1": self.expected_top1,
            "empirical_expected_entropy": self.empirical_expected_entropy,
            "empirical_expected_top1": self.empirical_expected_top1,
            "marginal_entropy": self.marginal_entropy,
            "marginal_top1": self.marginal_top1,
            "bidirectional": self.bidirectional,
        }

    def to_dependency(self, engine: QueryEngine) -> tuple[list[str], str]:
        lhs = [engine.schema.columns[col_id].name for col_id in self.lhs_cols]
        rhs = engine.schema.columns[self.rhs_col].name
        return lhs, rhs


class AFDSearcher:
    _GROUP_RE = re.compile(r"(?:^p(?P<prefix>\d+))|(?P<suffix>\d+)$")

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = validate_config(config or default_config())
        self.engine = QueryEngine(self.config)
        self.support: SupportEstimator = self.engine.support
        self._raw_candidate_cache: dict[tuple[int, tuple[int, ...]], CandidateScore | None] = {}
        self._candidate_cache: dict[tuple[int, tuple[int, ...]], CandidateScore | None] = {}
        self._reverse_cache: dict[tuple[int, tuple[int, ...]], float] = {}
        self._column_meta_cache: dict[int, dict[str, object]] = {}

    def _build_soft_contingency(self, rhs_col: int, support: SupportTable) -> list[dict[str, object]]:
        if self.config.search.export_soft_contingency_top_n <= 0 or support.num_rows == 0:
            return []

        evidences = support.iter_evidences()[: self.config.search.soft_contingency_max_rows]
        probabilities = self.engine.conditional_dist_batch(rhs_col, evidences)

        rows = []
        for row_id, evidence in enumerate(evidences):
            lhs_values = [self.engine.schema.decode_value(col_id, token_id) for col_id, token_id in evidence.items()]
            rhs_distribution = probabilities[row_id]
            top_indices = np.argsort(-rhs_distribution)[:3]
            rows.append(
                {
                    "lhs_values": lhs_values,
                    "p_x": float(support.probabilities[row_id]),
                    "empirical_weight": float(support.empirical_weights[row_id]),
                    "empirical_top1": None if support.empirical_top1 is None else float(support.empirical_top1[row_id]),
                    "top_rhs": [
                        {
                            "value": self.engine.schema.decode_value(rhs_col, int(index)),
                            "prob": float(rhs_distribution[index]),
                        }
                        for index in top_indices
                    ],
                }
            )
        return rows

    def _cache_key(self, rhs_col: int, lhs_cols: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
        return int(rhs_col), tuple(sorted(int(col_id) for col_id in lhs_cols))

    def _entropy_strength(self, entropies: np.ndarray | float, marginal_entropy: float, rhs_col: int) -> np.ndarray:
        rhs_vocab = max(int(self.engine.schema.columns[rhs_col].vocab_size), 2)
        fallback = max(float(np.log(rhs_vocab)), 1e-12)
        denom = max(float(marginal_entropy), fallback * 0.05, 1e-12)
        values = 1.0 - np.asarray(entropies, dtype=np.float64) / denom
        return np.clip(values, 0.0, 1.0)

    def _accuracy_strength(self, top1: np.ndarray | float, marginal_top1: float) -> np.ndarray:
        baseline = float(np.clip(marginal_top1, 0.0, 1.0))
        denom = max(1.0 - baseline, 1e-12)
        values = (np.asarray(top1, dtype=np.float64) - baseline) / denom
        return np.clip(values, 0.0, 1.0)

    def _column_meta(self, col_id: int) -> dict[str, object]:
        col_id = int(col_id)
        cached = self._column_meta_cache.get(col_id)
        if cached is not None:
            return cached

        column = self.engine.schema.columns[col_id]
        stats = dict(column.stats)
        policy = dict(stats.get("policy") or {})
        train_unique = float(stats.get("train_unique", 0.0) or 0.0)
        unique_ratio = float(policy.get("unique_ratio", 0.0) or 0.0)
        if unique_ratio <= 0.0:
            train_rows = max(float(self.engine.schema.train_rows), 1.0)
            unique_ratio = float(train_unique / train_rows) if train_unique > 0.0 else 0.0

        group_match = self._GROUP_RE.search(column.name)
        group_label = None
        if group_match is not None:
            group_label = group_match.group("prefix") or group_match.group("suffix")

        meta = {
            "name": column.name,
            "role": str(policy.get("role", "")),
            "unique_ratio": unique_ratio,
            "train_unique": int(train_unique),
            "group": group_label,
        }
        self._column_meta_cache[col_id] = meta
        return meta

    def _ramp(self, value: float, start: float, end: float) -> float:
        if end <= start:
            return 1.0 if value >= end else 0.0
        return float(np.clip((value - start) / (end - start), 0.0, 1.0))

    def _column_identifier_signal(self, col_id: int) -> float:
        meta = self._column_meta(col_id)
        role = str(meta["role"])
        if role in {"identifier", "quasi_identifier"}:
            return 1.0
        return self._ramp(
            float(meta["unique_ratio"]),
            self.config.search.lhs_identifier_start_ratio,
            self.config.search.lhs_identifier_full_ratio,
        )

    def _column_allowed(self, col_id: int, allowlist: tuple[str, ...], blocklist: tuple[str, ...]) -> bool:
        name = self.engine.schema.columns[int(col_id)].name
        if allowlist and not any(fnmatch.fnmatchcase(name, pattern) for pattern in allowlist):
            return False
        if blocklist and any(fnmatch.fnmatchcase(name, pattern) for pattern in blocklist):
            return False
        return True

    def _rhs_candidates(self) -> list[int]:
        return [
            col_id
            for col_id in self.engine.schema.searchable_rhs_indices()
            if self._column_allowed(col_id, self.config.search.rhs_allowlist, self.config.search.rhs_blocklist)
        ]

    def _lhs_candidates_for_rhs(self, rhs_col: int) -> list[int]:
        return [
            col_id
            for col_id in self.engine.schema.searchable_lhs_indices()
            if col_id != rhs_col
            and self._column_allowed(col_id, self.config.search.lhs_allowlist, self.config.search.lhs_blocklist)
        ]

    def _group_factor(self, lhs_cols: tuple[int, ...], rhs_col: int) -> float:
        mode = str(self.config.search.group_match_mode).strip().lower()
        if mode == "off":
            return 1.0

        rhs_group = self._column_meta(rhs_col).get("group")
        lhs_groups = {self._column_meta(col_id).get("group") for col_id in lhs_cols}
        lhs_groups.discard(None)
        if rhs_group is None or not lhs_groups:
            return 1.0
        if len(lhs_groups) == 1 and rhs_group in lhs_groups:
            return 1.0
        if mode == "hard":
            return 0.0
        return float(np.clip(self.config.search.cross_group_penalty, 0.0, 1.0))

    def _coverage_factor(self, support: SupportTable, weighted_non_null_ratio: float) -> float:
        row_factor = min(
            1.0,
            math.log1p(max(support.num_rows, 0)) / math.log1p(max(self.config.search.coverage_row_target, 1)),
        )
        effective_rows = int(np.count_nonzero(support.effective_counts > 0))
        effective_row_factor = min(
            1.0,
            math.log1p(max(effective_rows, 0))
            / math.log1p(max(self.config.search.coverage_effective_row_target, 1)),
        )
        mass_factor = min(
            1.0,
            float(support.effective_retained_mass) / max(float(self.config.search.coverage_mass_target), 1e-12),
        )
        signal = float(np.mean([row_factor, effective_row_factor, mass_factor, weighted_non_null_ratio]))
        penalty_weight = float(np.clip(self.config.search.coverage_penalty_weight, 0.0, 1.0))
        return float((1.0 - penalty_weight) + penalty_weight * signal)

    def _empirical_blend(self, lhs_cols: tuple[int, ...], rhs_col: int, coverage_factor: float) -> float:
        rhs_unique_ratio = float(self._column_meta(rhs_col)["unique_ratio"])
        rhs_high_card = self._ramp(
            rhs_unique_ratio,
            self.config.search.empirical_high_card_start_ratio,
            self.config.search.empirical_high_card_full_ratio,
        )
        rhs_low_card = 1.0 - self._ramp(
            rhs_unique_ratio,
            self.config.search.empirical_low_card_start_ratio,
            self.config.search.empirical_low_card_full_ratio,
        )
        rhs_card_support = math.sqrt(max(rhs_high_card, rhs_low_card))
        # Use mean for compound LHS: max() overstates when one identifier column
        # is mixed with non-identifier columns.  Mean gives a more accurate signal
        # for the overall LHS quality.
        _id_signals = [self._column_identifier_signal(col_id) for col_id in lhs_cols]
        lhs_identifier_signal = float(np.mean(_id_signals)) if _id_signals else 0.0
        base = float(np.clip(self.config.search.empirical_aux_weight, 0.0, 1.0))
        high_card_base = float(np.clip(self.config.search.empirical_high_card_base, 0.0, 1.0))
        bonus = float(np.clip(self.config.search.empirical_high_card_bonus, 0.0, 1.0))
        blend = base + high_card_base * rhs_card_support + bonus * rhs_card_support * lhs_identifier_signal
        # Identifier-direct bonus: when LHS is a near-unique identifier, empirical
        # signals are reliable regardless of RHS cardinality.  Identifiers
        # deterministically map to attributes in training data, so the empirical
        # accuracy is near-perfect.  Without this, low-cardinality RHS columns
        # (e.g. p1publisher with 6 values) get zero empirical weight even though
        # the identifier LHS guarantees correctness.
        if lhs_identifier_signal >= 0.65:
            identifier_direct = 0.65 * lhs_identifier_signal
            blend += identifier_direct
        blend *= float(np.clip(coverage_factor, 0.0, 1.0))
        return float(np.clip(blend, 0.0, self.config.search.max_empirical_blend))

    def _score_candidate_core(self, rhs_col: int, lhs_cols: tuple[int, ...]) -> CandidateScore | None:
        cache_key = self._cache_key(rhs_col, lhs_cols)
        if cache_key in self._raw_candidate_cache:
            return self._raw_candidate_cache[cache_key]

        support = self.support.build_support(cache_key[1], rhs_col)
        if support.num_rows == 0:
            self._raw_candidate_cache[cache_key] = None
            return None
        if support.retained_mass < self.config.search.min_retained_mass:
            self._raw_candidate_cache[cache_key] = None
            return None
        if support.effective_retained_mass < self.config.search.min_effective_retained_mass:
            self._raw_candidate_cache[cache_key] = None
            return None

        if self.config.search.exclude_rhs_special_tokens_for_scoring:
            summaries = self.engine.conditional_summary_batch_valid_rhs(rhs_col, support.iter_evidences())
        else:
            summaries = self.engine.conditional_summary_batch(rhs_col, support.iter_evidences())
        model_entropies = np.asarray([item.entropy for item in summaries], dtype=np.float64)
        model_top1 = np.asarray([item.top1_prob for item in summaries], dtype=np.float64)
        weights = support.probabilities

        expected_entropy = float(np.dot(weights, model_entropies))
        expected_top1 = float(np.dot(weights, model_top1))

        if self.config.search.exclude_rhs_special_tokens_for_scoring:
            marginal = self.engine.marginal_empirical_valid(rhs_col)
        else:
            marginal = self.engine.marginal_empirical(rhs_col)
        marginal_entropy = float(entropy_from_probs(marginal[None, :])[0])
        marginal_top1 = float(marginal.max())

        weighted_non_null_ratio = 1.0
        if support.non_null_ratios is not None:
            weighted_non_null_ratio = float(np.dot(weights, support.non_null_ratios))
        if weighted_non_null_ratio < self.config.search.min_weighted_non_null_ratio:
            self._raw_candidate_cache[cache_key] = None
            return None

        coverage_factor = self._coverage_factor(support, weighted_non_null_ratio)
        empirical_blend = self._empirical_blend(cache_key[1], rhs_col, coverage_factor)
        group_factor = self._group_factor(cache_key[1], rhs_col)
        if group_factor <= 0.0:
            self._raw_candidate_cache[cache_key] = None
            return None

        model_row_s_ent = self._entropy_strength(model_entropies, marginal_entropy, rhs_col)
        model_row_s_acc = self._accuracy_strength(model_top1, marginal_top1)
        model_s_ent = float(np.dot(weights, model_row_s_ent))
        model_s_acc = float(np.dot(weights, model_row_s_acc))
        model_score = float(
            self.config.search.score_alpha * model_s_ent + (1.0 - self.config.search.score_alpha) * model_s_acc
        )

        empirical_expected_entropy = expected_entropy
        empirical_expected_top1 = expected_top1
        empirical_row_s_ent = model_row_s_ent
        empirical_row_s_acc = model_row_s_acc
        if support.empirical_entropies is not None and support.empirical_top1 is not None:
            empirical_expected_entropy = float(np.dot(weights, support.empirical_entropies))
            empirical_expected_top1 = float(np.dot(weights, support.empirical_top1))
            empirical_row_s_ent = self._entropy_strength(support.empirical_entropies, marginal_entropy, rhs_col)
            empirical_row_s_acc = self._accuracy_strength(support.empirical_top1, marginal_top1)

        empirical_s_ent = float(np.dot(weights, empirical_row_s_ent))
        empirical_s_acc = float(np.dot(weights, empirical_row_s_acc))
        empirical_score = float(
            self.config.search.score_alpha * empirical_s_ent
            + (1.0 - self.config.search.score_alpha) * empirical_s_acc
        )

        base_empirical_weight = 1.0 - float(np.clip(self.config.search.model_score_weight, 0.0, 1.0))
        # Adaptive empirical boost: when empirical accuracy significantly exceeds
        # model accuracy, the model is likely failing due to [NULL] dominance in
        # high-null columns.  Increase empirical blend and reduce model weight
        # to let the reliable empirical signal compensate for the model's null bias.
        acc_gap = max(empirical_s_acc - model_s_acc, 0.0)
        effective_model_weight = float(np.clip(self.config.search.model_score_weight, 0.0, 1.0))
        effective_support = int(np.count_nonzero(support.effective_counts > 0)) if support.effective_counts is not None else support.num_rows
        if acc_gap > 0.15 and empirical_s_acc >= 0.90 and effective_support >= 10:
            gap_boost = min(acc_gap, 0.75)
            empirical_blend = float(np.clip(empirical_blend + gap_boost, 0.0, self.config.search.max_empirical_blend))
            # Reduce model weight when model is unreliable but empirical is strong
            effective_model_weight = max(effective_model_weight * 0.3, 0.10)
        base_empirical_weight = 1.0 - effective_model_weight
        empirical_bonus_weight = empirical_blend * base_empirical_weight
        # Asymmetric blend: full weight when empirical > model (boost),
        # reduced weight when empirical < model (correction for overconfidence).
        ent_delta = empirical_s_ent - model_s_ent
        acc_delta = empirical_s_acc - model_s_acc
        correction_weight = empirical_bonus_weight * 0.5  # dampened for negative deltas
        s_ent = float(model_s_ent + (empirical_bonus_weight if ent_delta >= 0 else correction_weight) * ent_delta)
        s_acc = float(model_s_acc + (empirical_bonus_weight if acc_delta >= 0 else correction_weight) * acc_delta)

        # Identifier-empirical override: when LHS is an identifier and empirical
        # signal is strong, let empirical score dominate.  Identifiers
        # deterministically map to attributes in training data, so the empirical
        # signal is ground truth.  The model may fail to learn this mapping due
        # to capacity or positional bias (e.g. p1-side in dblp10k).
        _id_signals_core = [self._column_identifier_signal(col_id) for col_id in lhs_cols]
        lhs_id_signal = float(np.mean(_id_signals_core)) if _id_signals_core else 0.0
        if lhs_id_signal >= 0.65 and empirical_s_acc >= 0.90:
            id_override = 0.70 * lhs_id_signal  # up to 0.70 boost toward empirical
            s_ent = float(np.clip(s_ent + id_override * max(empirical_s_ent - s_ent, 0.0), 0.0, 1.0))
            s_acc = float(np.clip(s_acc + id_override * max(empirical_s_acc - s_acc, 0.0), 0.0, 1.0))

        s_ent = float(np.clip(s_ent, 0.0, 1.0))
        s_acc = float(np.clip(s_acc, 0.0, 1.0))
        base_score = float(self.config.search.score_alpha * s_ent + (1.0 - self.config.search.score_alpha) * s_acc)
        score = float(np.clip(base_score * coverage_factor * group_factor, 0.0, 1.0))

        candidate = CandidateScore(
            rhs_col=rhs_col,
            lhs_cols=cache_key[1],
            s_ent=s_ent,
            s_acc=s_acc,
            score=score,
            model_score=model_score,
            empirical_score=empirical_score,
            empirical_blend=empirical_blend,
            empirical_bonus_weight=empirical_bonus_weight,
            coverage_factor=coverage_factor,
            group_factor=group_factor,
            model_s_ent=model_s_ent,
            model_s_acc=model_s_acc,
            empirical_s_ent=empirical_s_ent,
            empirical_s_acc=empirical_s_acc,
            reverse_score=0.0,
            direction_margin=0.0,
            support_rows=support.num_rows,
            effective_support_rows=int(np.count_nonzero(support.effective_counts > 0)),
            retained_mass=support.retained_mass,
            effective_retained_mass=support.effective_retained_mass,
            weighted_non_null_ratio=weighted_non_null_ratio,
            expected_entropy=expected_entropy,
            expected_top1=expected_top1,
            empirical_expected_entropy=empirical_expected_entropy,
            empirical_expected_top1=empirical_expected_top1,
            marginal_entropy=marginal_entropy,
            marginal_top1=marginal_top1,
        )
        self._raw_candidate_cache[cache_key] = candidate
        return candidate

    def _reverse_score(self, rhs_col: int, lhs_cols: tuple[int, ...]) -> float:
        cache_key = self._cache_key(rhs_col, lhs_cols)
        if cache_key in self._reverse_cache:
            return self._reverse_cache[cache_key]

        reverse_scores: list[float] = []
        for lhs_col in cache_key[1]:
            reverse_candidate = self._score_candidate_core(lhs_col, (rhs_col,))
            if reverse_candidate is not None:
                reverse_scores.append(reverse_candidate.model_s_acc)

        reverse_score = float(max(reverse_scores)) if reverse_scores else 0.0
        self._reverse_cache[cache_key] = reverse_score
        return reverse_score

    def _detect_bidirectional(
        self,
        discovered: list[CandidateScore],
    ) -> list[CandidateScore]:
        """Detect bidirectional FDs: if A->B was discovered, check if B->A should also be reported.

        For bidirectional FDs (e.g. AirportCode <-> AirportName), both directions have similar
        strength. The main search may miss one direction due to direction margin constraints.
        This method explicitly checks the reverse direction for discovered Level-1 FDs.
        """
        bidirectional_results: list[CandidateScore] = []
        # Only check Level-1 (single-column LHS) FDs for bidirectional
        level1_fds = [c for c in discovered if len(c.lhs_cols) == 1]

        for candidate in level1_fds:
            lhs_col = candidate.lhs_cols[0]
            rhs_col = candidate.rhs_col

            # Check if reverse direction (rhs_col -> lhs_col) was already discovered
            reverse_key = self._cache_key(lhs_col, (rhs_col,))
            already_in = any(
                c.rhs_col == lhs_col and c.lhs_cols == (rhs_col,)
                for c in discovered
            )
            if already_in:
                continue

            # Score the reverse direction
            reverse_candidate = self.score_candidate(lhs_col, (rhs_col,))
            if reverse_candidate is None:
                continue

            # Both directions must have high model_s_acc to qualify as bidirectional
            if (
                reverse_candidate.model_s_acc >= 0.85
                and candidate.model_s_acc >= 0.85
                and reverse_candidate.score >= self.config.search.min_score * 0.8
                and reverse_candidate.s_ent >= self.config.search.min_s_ent * 0.8
                and reverse_candidate.s_acc >= self.config.search.min_s_acc * 0.8
            ):
                reverse_candidate = CandidateScore(
                    rhs_col=reverse_candidate.rhs_col,
                    lhs_cols=reverse_candidate.lhs_cols,
                    s_ent=reverse_candidate.s_ent,
                    s_acc=reverse_candidate.s_acc,
                    score=reverse_candidate.score,
                    model_score=reverse_candidate.model_score,
                    empirical_score=reverse_candidate.empirical_score,
                    empirical_blend=reverse_candidate.empirical_blend,
                    empirical_bonus_weight=reverse_candidate.empirical_bonus_weight,
                    coverage_factor=reverse_candidate.coverage_factor,
                    group_factor=reverse_candidate.group_factor,
                    model_s_ent=reverse_candidate.model_s_ent,
                    model_s_acc=reverse_candidate.model_s_acc,
                    empirical_s_ent=reverse_candidate.empirical_s_ent,
                    empirical_s_acc=reverse_candidate.empirical_s_acc,
                    reverse_score=reverse_candidate.reverse_score,
                    direction_margin=reverse_candidate.direction_margin,
                    support_rows=reverse_candidate.support_rows,
                    effective_support_rows=reverse_candidate.effective_support_rows,
                    retained_mass=reverse_candidate.retained_mass,
                    effective_retained_mass=reverse_candidate.effective_retained_mass,
                    weighted_non_null_ratio=reverse_candidate.weighted_non_null_ratio,
                    expected_entropy=reverse_candidate.expected_entropy,
                    expected_top1=reverse_candidate.expected_top1,
                    empirical_expected_entropy=reverse_candidate.empirical_expected_entropy,
                    empirical_expected_top1=reverse_candidate.empirical_expected_top1,
                    marginal_entropy=reverse_candidate.marginal_entropy,
                    marginal_top1=reverse_candidate.marginal_top1,
                    bidirectional=True,
                )
                bidirectional_results.append(reverse_candidate)

        return bidirectional_results

    def score_candidate(self, rhs_col: int, lhs_cols: tuple[int, ...]) -> CandidateScore | None:
        cache_key = self._cache_key(rhs_col, lhs_cols)
        if cache_key in self._candidate_cache:
            return self._candidate_cache[cache_key]

        core_candidate = self._score_candidate_core(rhs_col, lhs_cols)
        if core_candidate is None:
            self._candidate_cache[cache_key] = None
            return None

        reverse_score = self._reverse_score(rhs_col, lhs_cols)
        candidate = CandidateScore(
            rhs_col=core_candidate.rhs_col,
            lhs_cols=core_candidate.lhs_cols,
            s_ent=core_candidate.s_ent,
            s_acc=core_candidate.s_acc,
            score=core_candidate.score,
            model_score=core_candidate.model_score,
            empirical_score=core_candidate.empirical_score,
            empirical_blend=core_candidate.empirical_blend,
            empirical_bonus_weight=core_candidate.empirical_bonus_weight,
            coverage_factor=core_candidate.coverage_factor,
            group_factor=core_candidate.group_factor,
            model_s_ent=core_candidate.model_s_ent,
            model_s_acc=core_candidate.model_s_acc,
            empirical_s_ent=core_candidate.empirical_s_ent,
            empirical_s_acc=core_candidate.empirical_s_acc,
            reverse_score=reverse_score,
            direction_margin=float(core_candidate.model_s_acc - reverse_score),
            support_rows=core_candidate.support_rows,
            effective_support_rows=core_candidate.effective_support_rows,
            retained_mass=core_candidate.retained_mass,
            effective_retained_mass=core_candidate.effective_retained_mass,
            weighted_non_null_ratio=core_candidate.weighted_non_null_ratio,
            expected_entropy=core_candidate.expected_entropy,
            expected_top1=core_candidate.expected_top1,
            empirical_expected_entropy=core_candidate.empirical_expected_entropy,
            empirical_expected_top1=core_candidate.empirical_expected_top1,
            marginal_entropy=core_candidate.marginal_entropy,
            marginal_top1=core_candidate.marginal_top1,
        )
        self._candidate_cache[cache_key] = candidate
        return candidate

    def _passes_thresholds(self, candidate: CandidateScore | None) -> bool:
        if candidate is None:
            return False

        # Relax direction margin when reverse score is also high (bidirectional FD candidate).
        # For bidirectional FDs (e.g. AirportCode <-> AirportName), both directions have similar
        # strength, so the margin is naturally small. We use a relaxed threshold in this case.
        effective_min_margin = self.config.search.min_direction_margin

        mode = str(self.config.search.direction_margin_mode).strip().lower()
        if mode == "off":
            effective_min_margin = -1.0
        elif mode == "single_only" and len(candidate.lhs_cols) > 1:
            # Multi-column LHS: relax direction margin instead of skipping entirely.
            # The true reverse for A,B->C is C->A,B, which is expensive to compute.
            # Apply a relaxed threshold — minimality filter and delta_gain provide
            # the primary redundancy control, but a very negative margin is still
            # a warning sign.
            effective_min_margin = max(effective_min_margin * 2.0, -0.20)
        else:
            # Single-column LHS (or mode=="all"): keep existing bidirectional relaxation
            # High-null column relaxation: when empirical accuracy is strong but model
            # accuracy is moderate (model learned [NULL] bias), the direction margin
            # is unreliable because the model can't predict the sparse non-null values.
            # This must be checked BEFORE reverse_score relaxation, because the reverse
            # score is inflated for high-null columns (reverse direction has good model
            # accuracy while forward direction doesn't).
            if candidate.empirical_s_acc >= 0.90 and candidate.model_s_acc < 0.60:
                effective_min_margin = -1.0
            elif candidate.reverse_score >= 0.99:
                effective_min_margin = -1.0
            elif candidate.reverse_score >= 0.85:
                effective_min_margin *= 0.3
            elif candidate.empirical_s_acc >= 0.90 and candidate.model_s_acc < 0.15:
                effective_min_margin = -1.0

        return (
            candidate.s_ent >= self.config.search.min_s_ent
            and candidate.s_acc >= self.config.search.min_s_acc
            and candidate.score >= self.config.search.min_score
            and candidate.direction_margin >= effective_min_margin
            and candidate.support_rows > 0
            and candidate.retained_mass >= self.config.search.min_retained_mass
            and candidate.effective_retained_mass >= self.config.search.min_effective_retained_mass
            and candidate.weighted_non_null_ratio >= self.config.search.min_weighted_non_null_ratio
            and candidate.group_factor > 0.0
        )

    def _approx_unique_penalty(self, lhs_cols: tuple[int, ...]) -> float:
        """如果 LHS 包含近似唯一列且与其他列组合，施加冗余惩罚。"""
        if len(lhs_cols) <= 1:
            return 1.0
        for col_id in lhs_cols:
            unique_ratio = float(self._column_meta(col_id)["unique_ratio"])
            if unique_ratio >= 0.95:
                return 0.6
        return 1.0

    def _minimality_filter(self, candidates: list[CandidateScore]) -> list[CandidateScore]:
        # 按 RHS 分组，每组内按子集大小升序、score 降序排序，确保小集合先被检查
        by_rhs: dict[int, list[CandidateScore]] = {}
        for c in candidates:
            by_rhs.setdefault(c.rhs_col, []).append(c)

        kept: list[CandidateScore] = []
        for rhs_col, rhs_candidates in by_rhs.items():
            rhs_candidates.sort(key=lambda c: (len(c.lhs_cols), -c.score, -c.s_ent, -c.s_acc))
            accepted_for_rhs: list[CandidateScore] = []
            for candidate in rhs_candidates:
                lhs_set = set(candidate.lhs_cols)
                redundant = False
                # Require larger gain for compound LHS containing high-cardinality columns.
                # Such compounds are often redundant (e.g. CloseAmount,Status -> Disposition
                # where Status alone suffices). High-cardinality columns like CloseAmount
                # (unique_ratio=0.15) add noise rather than signal.
                effective_delta = self.config.search.delta_gain
                if len(candidate.lhs_cols) > 1:
                    for col_id in candidate.lhs_cols:
                        if self._column_meta(col_id)["unique_ratio"] > 0.01:
                            effective_delta = max(effective_delta, 0.08)
                            break
                for prev in accepted_for_rhs:
                    if set(prev.lhs_cols).issubset(lhs_set):
                        abs_gain = candidate.score - prev.score
                        if abs_gain <= effective_delta:
                            redundant = True
                            break
                        # When the accepted subset is already very strong
                        # (score ≥ 0.95), require a stricter absolute gain
                        # of at least 0.05.  A compound that only marginally
                        # improves an already-high score is almost certainly
                        # redundant (e.g. education_num,hours_per_week→education
                        # gaining 0.04 over education_num→education at 0.959).
                        if prev.score >= 0.95 and abs_gain < 0.05:
                            redundant = True
                            break
                if not redundant:
                    accepted_for_rhs.append(candidate)
            kept.extend(accepted_for_rhs)

        kept.sort(key=lambda item: (item.score, item.s_ent, item.s_acc, item.retained_mass), reverse=True)
        return kept

    def search_rhs(self, rhs_col: int) -> list[CandidateScore]:
        if not self._column_allowed(rhs_col, self.config.search.rhs_allowlist, self.config.search.rhs_blocklist):
            return []
        lhs_candidates = self._lhs_candidates_for_rhs(rhs_col)
        scored_level1: list[CandidateScore] = []
        final_results: list[CandidateScore] = []

        for lhs_col in lhs_candidates:
            candidate = self.score_candidate(rhs_col, (lhs_col,))
            if candidate is None:
                continue
            scored_level1.append(candidate)
            if self._passes_thresholds(candidate):
                final_results.append(candidate)

        scored_level1.sort(key=lambda item: (item.score, item.s_ent, item.s_acc, item.retained_mass), reverse=True)
        if self.config.search.max_lhs_size <= 1:
            return self._minimality_filter(final_results)

        # CAFD-style 自适应剪枝：根据数据集标识符强度动态调整阈值
        # 低标识符强度数据集（如 dblp10k）需要保留更多候选以发现多列组合 FD
        max_identifier_signal = max(
            (self._column_identifier_signal(col_id) for col_id in lhs_candidates), default=0.0
        )
        pruning_threshold = max(
            self.config.search.min_s_acc * (0.25 + 0.25 * max_identifier_signal), 0.03
        )
        pruned_lhs_set = frozenset(
            c.lhs_cols[0] for c in scored_level1 if c.model_s_acc >= pruning_threshold
        )
        # 如果剪枝后剩余列太少，回退到保留前 80% 的列
        if len(pruned_lhs_set) < max(2, len(lhs_candidates) // 5):
            keep_count = max(2, int(len(scored_level1) * 0.8))
            pruned_lhs_set = frozenset(c.lhs_cols[0] for c in scored_level1[:keep_count])

        # Recursive beam expansion up to max_lhs_size
        beam: list[tuple[tuple[int, ...], float]] = [
            (c.lhs_cols, c.score)
            for c in scored_level1[: self.config.search.level1_top_k]
            if c.lhs_cols[0] in pruned_lhs_set
        ]

        seen_lhs: set[tuple[int, ...]] = {c.lhs_cols for c in scored_level1}

        for _depth in range(self.config.search.max_lhs_size - 1):
            if not beam:
                break
            next_beam: list[tuple[tuple[int, ...], float]] = []
            for current_lhs, best_subset_score in beam:
                if len(current_lhs) >= self.config.search.max_lhs_size:
                    continue
                for extra_col in lhs_candidates:
                    if extra_col not in pruned_lhs_set:
                        continue
                    if extra_col in current_lhs:
                        continue
                    new_lhs = tuple(sorted(current_lhs + (extra_col,)))
                    if new_lhs in seen_lhs:
                        continue
                    seen_lhs.add(new_lhs)

                    new_candidate = self.score_candidate(rhs_col, new_lhs)
                    if not self._passes_thresholds(new_candidate):
                        continue
                    assert new_candidate is not None

                    # 近似唯一 LHS 冗余惩罚
                    penalty = self._approx_unique_penalty(new_lhs)
                    effective_score = new_candidate.score * penalty
                    effective_best = best_subset_score * self._approx_unique_penalty(current_lhs)
                    gain = effective_score - effective_best
                    if gain < self.config.search.delta_gain:
                        continue
                    final_results.append(new_candidate)
                    next_beam.append((new_lhs, new_candidate.score))

            next_beam.sort(key=lambda x: x[1], reverse=True)
            beam = next_beam[: self.config.search.level1_top_k]

        return self._minimality_filter(final_results)

    def run(self, rhs_limit: int | None = None) -> dict[str, object]:
        start_time = time.time()
        results: list[dict[str, object]] = []
        discovered_dependencies: list[tuple[list[str], str]] = []
        rhs_columns = self._rhs_candidates()
        if rhs_limit is not None:
            rhs_columns = rhs_columns[:rhs_limit]

        summary = {
            "rhs_processed": 0,
            "candidates_kept": 0,
            "results": {},
            "method": {
                "lhs_size_limit": self.config.search.max_lhs_size,
                "scoring": "model_s_func_plus_model_s_ent_with_positive_empirical_bonus",
                "direction_margin": "model_s_acc_forward_minus_max_reverse_model_s_acc",
                "support_probability": "empirical_mass_with_optional_independence_shrinkage",
            },
            "config": {
                "dataset": self.config.paths.dataset_name,
                "search": self.config.search.__dict__,
                "data": self.config.data.__dict__,
                "model": self.config.model.__dict__,
                "training": self.config.training.__dict__,
                "calibration": self.config.calibration.__dict__,
            },
        }

        if self.config.paths.preprocess_summary_path.exists():
            with self.config.paths.preprocess_summary_path.open("r", encoding="utf-8") as handle:
                summary["preprocess_summary"] = json.load(handle)

        all_candidate_scores: list[CandidateScore] = []
        for rhs_col in rhs_columns:
            rhs_name = self.engine.schema.columns[rhs_col].name
            candidates = self.search_rhs(rhs_col)
            all_candidate_scores.extend(candidates)
            summary["rhs_processed"] += 1
            summary["candidates_kept"] += len(candidates)
            summary["results"][rhs_name] = [candidate.to_dict(self.engine) for candidate in candidates]

            for candidate in candidates:
                record = candidate.to_dict(self.engine)
                discovered_dependencies.append(candidate.to_dependency(self.engine))
                if self.config.search.export_soft_contingency_top_n > 0:
                    support = self.support.build_support(candidate.lhs_cols, candidate.rhs_col)
                    record["soft_contingency_preview"] = self._build_soft_contingency(candidate.rhs_col, support)
                results.append(record)

        # Bidirectional FD detection: for discovered Level-1 FDs, check if the reverse
        # direction should also be reported (e.g. AirportCode <-> AirportName).
        bidirectional_new = self._detect_bidirectional(all_candidate_scores)
        for bi_candidate in bidirectional_new:
            bi_name = self.engine.schema.columns[bi_candidate.rhs_col].name
            bi_record = bi_candidate.to_dict(self.engine)
            summary["results"].setdefault(bi_name, []).append(bi_record)
            summary["candidates_kept"] += 1
            discovered_dependencies.append(bi_candidate.to_dependency(self.engine))
            results.append(bi_record)

        unique_dependencies = sorted(
            {(tuple(sorted(lhs)), rhs) for lhs, rhs in discovered_dependencies},
            key=lambda item: (item[0], item[1]),
        )
        discovered_dependency_list = [(list(lhs), rhs) for lhs, rhs in unique_dependencies]

        precision = 0.0
        recall = 0.0
        f1 = 0.0
        if self.config.paths.groundtruth_path.exists():
            precision, recall, f1 = evaluate_fd(
                discovered_fds=discovered_dependency_list,
                ground_truth_path=str(self.config.paths.groundtruth_path),
            )

        elapsed_ms = int((time.time() - start_time) * 1000)
        write_discovered_report(
            discovered_fds=discovered_dependency_list,
            output_path=str(self.config.paths.discovered_report_path),
            precision=precision,
            recall=recall,
            f1=f1,
            elapsed_ms=elapsed_ms,
        )

        summary["discovered_fd_count"] = len(discovered_dependency_list)
        summary["precision"] = precision
        summary["recall"] = recall
        summary["f1"] = f1
        summary["runtime_ms"] = elapsed_ms
        summary["groundtruth_path"] = str(self.config.paths.groundtruth_path)
        summary["report_path"] = str(self.config.paths.discovered_report_path)
        summary["discovered_fds"] = [
            {
                "lhs": lhs,
                "rhs": rhs,
            }
            for lhs, rhs in discovered_dependency_list
        ]

        with self.config.paths.search_results_path.open("w", encoding="utf-8") as handle:
            for record in results:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        with self.config.paths.search_summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=True, indent=2)
        return summary


def run_search(config: PipelineConfig | None = None, rhs_limit: int | None = None) -> dict[str, object]:
    searcher = AFDSearcher(validate_config(config or default_config()))
    return searcher.run(rhs_limit=rhs_limit)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AR conditional FD search.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME, choices=SUPPORTED_DATASETS, help="Dataset name.")
    parser.add_argument("--rhs-limit", type=int, default=None, help="Only score the first N RHS columns.")
    parser.add_argument("--support-beta", type=float, default=None, help="Override support shrinkage beta.")
    parser.add_argument("--min-support-count", type=int, default=None, help="Override minimum support count.")
    parser.add_argument("--min-s-ent", type=float, default=None, help="Override entropy-score threshold.")
    parser.add_argument("--min-s-acc", type=float, default=None, help="Override accuracy-score threshold.")
    parser.add_argument("--min-score", type=float, default=None, help="Override blended score threshold.")
    parser.add_argument("--score-alpha", type=float, default=None, help="Override entropy/top1 blend weight.")
    parser.add_argument("--direction-margin", type=float, default=None, help="Override minimum direction margin.")
    parser.add_argument("--delta-gain", type=float, default=None, help="Override minimum gain over subsets.")
    parser.add_argument("--max-lhs-size", type=int, default=None, help="Override maximum lhs size.")
    parser.add_argument("--top-k", type=int, default=None, help="Override level-1 beam width.")
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional row cap for smoke tests.")
    parser.add_argument("--continuous-bins", type=int, default=None, help="Override bucket count.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = default_config(args.dataset)
    if args.sample_rows is not None:
        config.data.sample_rows = args.sample_rows
    if args.continuous_bins is not None:
        config.data.continuous_bins = args.continuous_bins
    if args.support_beta is not None:
        config.search.support_beta = args.support_beta
    if args.min_support_count is not None:
        config.search.min_support_count = args.min_support_count
    if args.min_s_ent is not None:
        config.search.min_s_ent = args.min_s_ent
    if args.min_s_acc is not None:
        config.search.min_s_acc = args.min_s_acc
    if args.min_score is not None:
        config.search.min_score = args.min_score
    if args.score_alpha is not None:
        config.search.score_alpha = args.score_alpha
    if args.direction_margin is not None:
        config.search.min_direction_margin = args.direction_margin
    if args.delta_gain is not None:
        config.search.delta_gain = args.delta_gain
    if args.max_lhs_size is not None:
        config.search.max_lhs_size = args.max_lhs_size
    if args.top_k is not None:
        config.search.level1_top_k = args.top_k
    validate_config(config)
    summary = run_search(config, rhs_limit=args.rhs_limit)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
