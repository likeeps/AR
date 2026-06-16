from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from AR.config import ModelConfig, PipelineConfig, default_config
from AR.data import load_artifacts
from AR.masking import apply_mask_tokens
from AR.metrics import entropy_from_probs
from AR.model import AnyOrderConditionalTransformer
from AR.preprocess import canonicalize_discrete_numeric, canonicalize_text, parse_continuous, run_preprocessing
from AR.schema import DatasetSchema
from AR.support import SupportEstimator
from AR.train import choose_device


@dataclass
class QuerySummary:
    top1_prob: float
    entropy: float


class SummaryCache:
    def __init__(self, max_items: int = 50000) -> None:
        self.max_items = max_items
        self.data: OrderedDict[tuple[int, tuple[tuple[int, int], ...]], QuerySummary] = OrderedDict()

    def get(self, key: tuple[int, tuple[tuple[int, int], ...]]) -> QuerySummary | None:
        if key not in self.data:
            return None
        value = self.data.pop(key)
        self.data[key] = value
        return value

    def put(self, key: tuple[int, tuple[tuple[int, int], ...]], value: QuerySummary) -> None:
        if key in self.data:
            self.data.pop(key)
        self.data[key] = value
        if len(self.data) > self.max_items:
            self.data.popitem(last=False)


class QueryEngine:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        device: str | None = None,
    ) -> None:
        self.config = config or default_config()
        self.config.paths.ensure_dirs()
        run_preprocessing(self.config, force=False)
        artifacts = load_artifacts(self.config, mmap_train=True)
        self.schema: DatasetSchema = artifacts.schema
        self.support = SupportEstimator(artifacts.train_tokens, artifacts.schema, self.config.search)

        requested_device = device or self.config.training.device
        self.device = choose_device(requested_device)

        checkpoint = torch.load(self.config.paths.best_checkpoint_path, map_location=self.device)
        # Use model config from checkpoint if available (handles architecture changes gracefully).
        if "model_config" in checkpoint:
            model_config = ModelConfig(**checkpoint["model_config"])
        else:
            model_config = self.config.model
        self.model = AnyOrderConditionalTransformer(self.schema, model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.temperatures = self._load_temperatures()
        self.summary_cache = SummaryCache()

    def _load_temperatures(self) -> dict[str, float]:
        if not self.config.paths.temperature_path.exists():
            return {column.name: 1.0 for column in self.schema.columns}
        with self.config.paths.temperature_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {str(key): float(value) for key, value in dict(payload.get("temperatures", {})).items()}

    def _normalize_evidence_token(self, col_id: int, value: Any) -> int:
        column = self.schema.columns[col_id]
        if isinstance(value, (np.integer, int)):
            return int(value)
        if column.column_type == "categorical":
            token = canonicalize_text(value)
        elif column.column_type == "discrete_numeric":
            token = canonicalize_discrete_numeric(value)
        elif column.column_type == "continuous_bucket":
            parsed = parse_continuous(value)
            if parsed is None:
                return column.null_id
            edges = np.asarray(column.bucket_edges, dtype=np.float64)
            bucket_id = int(np.searchsorted(edges[1:-1], parsed, side="right"))
            token = f"bin_{bucket_id:02d}"
        else:
            raise ValueError(f"Unsupported column type: {column.column_type}")
        return column.encode_token(token)

    def _normalize_evidence(self, evidence: dict[int | str, Any]) -> dict[int, int]:
        normalized: dict[int, int] = {}
        for key, value in evidence.items():
            col_id = self.schema.column_index(key)
            normalized[col_id] = self._normalize_evidence_token(col_id, value)
        return normalized

    def _cache_key(self, rhs_col: int, evidence: dict[int, int]) -> tuple[int, tuple[tuple[int, int], ...]]:
        return rhs_col, tuple(sorted((int(col_id), int(token_id)) for col_id, token_id in evidence.items()))

    def _build_batch(self, normalized_evidences: list[dict[int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(normalized_evidences)
        num_columns = self.schema.num_columns

        token_matrix = np.tile(np.asarray(self.schema.mask_token_ids, dtype=np.int64), (batch_size, 1))
        observed_matrix = np.zeros((batch_size, num_columns), dtype=bool)

        for row_id, evidence in enumerate(normalized_evidences):
            for col_id, token_id in evidence.items():
                token_matrix[row_id, int(col_id)] = int(token_id)
                observed_matrix[row_id, int(col_id)] = True

        token_tensor = torch.from_numpy(token_matrix).to(self.device)
        observed_tensor = torch.from_numpy(observed_matrix).to(self.device)
        return token_tensor, observed_tensor

    @torch.no_grad()
    def conditional_dist_batch(
        self,
        rhs_col: int | str,
        evidences: list[dict[int | str, Any]],
    ) -> np.ndarray:
        rhs_id = self.schema.column_index(rhs_col)
        normalized = [self._normalize_evidence(evidence) for evidence in evidences]
        token_tensor, observed_tensor = self._build_batch(normalized)
        masked_inputs = apply_mask_tokens(token_tensor, observed_tensor, self.model.mask_token_ids)
        hidden = self.model.encode(masked_inputs, observed_tensor)
        logits = self.model.column_logits(hidden, rhs_id)
        temperature = self.temperatures.get(self.schema.columns[rhs_id].name, 1.0)
        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        return probs.cpu().numpy()

    @torch.no_grad()
    def conditional_summary_batch(
        self,
        rhs_col: int | str,
        evidences: list[dict[int | str, Any]],
    ) -> list[QuerySummary]:
        rhs_id = self.schema.column_index(rhs_col)
        normalized = [self._normalize_evidence(evidence) for evidence in evidences]

        pending_indices: list[int] = []
        pending_evidences: list[dict[int, int]] = []
        results: list[QuerySummary | None] = [None] * len(normalized)

        for idx, evidence in enumerate(normalized):
            key = self._cache_key(rhs_id, evidence)
            cached = self.summary_cache.get(key)
            if cached is not None:
                results[idx] = cached
            else:
                pending_indices.append(idx)
                pending_evidences.append(evidence)

        if pending_evidences:
            for start in range(0, len(pending_evidences), self.config.search.query_batch_size):
                stop = start + self.config.search.query_batch_size
                chunk = pending_evidences[start:stop]
                token_tensor, observed_tensor = self._build_batch(chunk)
                masked_inputs = apply_mask_tokens(token_tensor, observed_tensor, self.model.mask_token_ids)
                hidden = self.model.encode(masked_inputs, observed_tensor)
                logits = self.model.column_logits(hidden, rhs_id)
                temperature = self.temperatures.get(self.schema.columns[rhs_id].name, 1.0)
                probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1).cpu().numpy()
                entropies = entropy_from_probs(probs)
                top1_probs = probs.max(axis=1)

                for local_idx, evidence in enumerate(chunk):
                    global_idx = pending_indices[start + local_idx]
                    summary = QuerySummary(
                        top1_prob=float(top1_probs[local_idx]),
                        entropy=float(entropies[local_idx]),
                    )
                    results[global_idx] = summary
                    self.summary_cache.put(self._cache_key(rhs_id, evidence), summary)

        return [item for item in results if item is not None]

    @torch.no_grad()
    def conditional_summary_batch_valid_rhs(
        self,
        rhs_col: int | str,
        evidences: list[dict[int | str, Any]],
    ) -> list[QuerySummary]:
        """Like conditional_summary_batch but entropy/top1 computed over non-special tokens only."""
        rhs_id = self.schema.column_index(rhs_col)
        valid_mask = self.support.valid_rhs_mask(rhs_id)
        normalized = [self._normalize_evidence(evidence) for evidence in evidences]

        pending_indices: list[int] = []
        pending_evidences: list[dict[int, int]] = []
        results: list[QuerySummary | None] = [None] * len(normalized)

        for idx, evidence in enumerate(normalized):
            pending_indices.append(idx)
            pending_evidences.append(evidence)

        if pending_evidences:
            for start in range(0, len(pending_evidences), self.config.search.query_batch_size):
                stop = start + self.config.search.query_batch_size
                chunk = pending_evidences[start:stop]
                token_tensor, observed_tensor = self._build_batch(chunk)
                masked_inputs = apply_mask_tokens(token_tensor, observed_tensor, self.model.mask_token_ids)
                hidden = self.model.encode(masked_inputs, observed_tensor)
                logits = self.model.column_logits(hidden, rhs_id)
                temperature = self.temperatures.get(self.schema.columns[rhs_id].name, 1.0)
                probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1).cpu().numpy()

                # Mask special tokens and renormalize
                probs[:, ~valid_mask] = 0.0
                row_sums = probs.sum(axis=1, keepdims=True)
                row_sums = np.maximum(row_sums, 1e-12)
                probs = probs / row_sums

                entropies = entropy_from_probs(probs)
                top1_probs = probs.max(axis=1)

                for local_idx, evidence in enumerate(chunk):
                    global_idx = pending_indices[start + local_idx]
                    summary = QuerySummary(
                        top1_prob=float(top1_probs[local_idx]),
                        entropy=float(entropies[local_idx]),
                    )
                    results[global_idx] = summary

        return [item for item in results if item is not None]

    def conditional_dist(self, rhs_col: int | str, evidence: dict[int | str, Any]) -> np.ndarray:
        return self.conditional_dist_batch(rhs_col, [evidence])[0]

    def conditional_top1_prob(self, rhs_col: int | str, evidence: dict[int | str, Any]) -> float:
        return self.conditional_summary_batch(rhs_col, [evidence])[0].top1_prob

    def conditional_entropy(self, rhs_col: int | str, evidence: dict[int | str, Any]) -> float:
        return self.conditional_summary_batch(rhs_col, [evidence])[0].entropy

    def marginal_empirical(self, rhs_col: int | str) -> np.ndarray:
        return self.support.marginal_distribution(self.schema.column_index(rhs_col))

    def marginal_empirical_valid(self, rhs_col: int | str) -> np.ndarray:
        """Marginal distribution over non-special RHS tokens, renormalized."""
        return self.support.marginal_distribution_valid(self.schema.column_index(rhs_col))

    def support_prob(self, lhs_cols: tuple[int | str, ...], lhs_vals: tuple[int | str, ...]) -> float:
        lhs_ids = tuple(sorted(self.schema.column_index(column) for column in lhs_cols))
        encoded_vals = []
        for col_id, value in zip(lhs_ids, lhs_vals):
            encoded_vals.append(self._normalize_evidence_token(col_id, value))
        return self.support.support_probability(lhs_ids, tuple(encoded_vals))
