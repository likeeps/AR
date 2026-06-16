from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from AR.config import DEFAULT_DATASET_NAME, PipelineConfig, SPECIAL_TOKENS, SUPPORTED_DATASETS, default_config
from AR.datasets import DatasetRuntime, load_runtime_dataset
from AR.schema import ColumnSchema, DatasetSchema


PREPROCESS_VERSION = 3
_MULTISPACE_RE = re.compile(r"\s+")


def _is_nullish(value: object) -> bool:
    return value is None or (isinstance(value, (float, np.floating)) and math.isnan(float(value)))


def canonicalize_text(value: object) -> str:
    if _is_nullish(value):
        return "[NULL]"
    text = unicodedata.normalize("NFKC", str(value))
    text = text.strip().lower()
    text = text.replace("`", "'")
    text = _MULTISPACE_RE.sub(" ", text)
    return text if text else "[NULL]"


def canonicalize_discrete_numeric(value: object) -> str:
    if _is_nullish(value):
        return "[NULL]"
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return "[NULL]"
    return str(numeric)


def parse_continuous(value: object) -> float | None:
    if _is_nullish(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class SplitArrays:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


class DatasetPreprocessor:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.data_config = config.data

    def load_runtime(self) -> DatasetRuntime:
        return load_runtime_dataset(
            self.config.paths.dataset_name,
            self.config.paths.repo_root_path,
            sample_rows=self.data_config.sample_rows,
        )

    def _existing_artifacts_match_config(self) -> bool:
        summary_path = self.config.paths.preprocess_summary_path
        if not summary_path.exists():
            return False
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        return (
            int(summary.get("preprocess_version", -1)) == PREPROCESS_VERSION
            and str(summary.get("dataset_name", "")).lower() == self.config.paths.dataset_name
            and int(summary.get("continuous_bins", -1)) == int(self.data_config.continuous_bins)
            and int(summary.get("rare_token_min_freq", -1)) == int(self.data_config.rare_token_min_freq)
            and summary.get("sample_rows") == self.data_config.sample_rows
            and str(summary.get("search_space_mode", "balanced")).strip().lower() == self._search_space_mode()
        )

    def split_array(self, array: np.ndarray) -> SplitArrays:
        total = int(array.shape[0])
        rng = np.random.default_rng(self.data_config.seed)
        order = rng.permutation(total)

        train_end = int(total * self.data_config.train_ratio)
        val_end = train_end + int(total * self.data_config.val_ratio)
        train_ids = order[:train_end]
        val_ids = order[train_end:val_end]
        test_ids = order[val_end:]

        train = np.asarray(array[train_ids])
        val = np.asarray(array[val_ids])
        test = np.asarray(array[test_ids])
        return SplitArrays(train=train, val=val, test=test)

    def _build_vocab(
        self,
        values: list[str],
        rare_token_min_freq: int,
    ) -> tuple[list[str], Counter[str]]:
        counts = Counter(values)
        kept_tokens: list[str] = []

        for token, count in counts.most_common():
            if token in {"[NULL]", "[UNK]", "[MASK]", "[RARE]"}:
                continue
            if rare_token_min_freq > 1 and count < rare_token_min_freq:
                continue
            kept_tokens.append(token)

        vocab = list(SPECIAL_TOKENS) + kept_tokens
        return vocab, counts

    def _categorical_token_from_raw(
        self,
        runtime: DatasetRuntime,
        column_name: str,
        raw_value: object,
    ) -> str:
        if _is_nullish(raw_value):
            return "[NULL]"

        inverse_map = runtime.inverse_category_maps.get(column_name)
        if inverse_map:
            try:
                encoded = int(float(raw_value))
            except (TypeError, ValueError):
                return canonicalize_text(raw_value)
            if encoded < 0:
                return "[NULL]"
            decoded = inverse_map.get(encoded)
            if decoded is None:
                return "[UNK]"
            return canonicalize_text(decoded)

        return canonicalize_text(raw_value)

    def _column_model_type(self, runtime: DatasetRuntime, column_name: str, policy) -> str:
        if runtime.is_categorical(column_name) or policy.family == "categorical":
            return "categorical"
        if runtime.is_continuous(column_name) and policy.analysis_mode != "exact":
            return "continuous_bucket"
        if policy.analysis_mode == "binned":
            return "continuous_bucket"
        return "discrete_numeric"

    def _search_space_mode(self) -> str:
        mode = str(getattr(self.config.search, "search_space_mode", "balanced") or "balanced").strip().lower()
        if mode not in {"strict", "balanced", "permissive"}:
            return "balanced"
        return mode

    def _policy_searchable_flags(self, policy) -> tuple[bool, bool]:
        return policy.analysis_mode in {"exact", "binned"}, policy.rhs_mode == "exact"

    def _is_constant_like(self, policy) -> bool:
        return str(getattr(policy, "role", "")) == "constant" or int(getattr(policy, "unique_count", 0)) <= 1

    def _searchable_flags(
        self,
        runtime: DatasetRuntime,
        column_name: str,
        policy,
        model_type: str,
    ) -> tuple[bool, bool]:
        if runtime.is_target(column_name):
            return False, False
        if self._is_constant_like(policy):
            return False, False

        role = str(getattr(policy, "role", ""))
        # Identifier-like columns (unique keys, IDs) must never appear as RHS:
        # they are near-unique so any LHS trivially determines them, producing spurious FDs.
        if role == "near_identifier":
            if self._search_space_mode() == "permissive":
                # In permissive mode, allow near_identifier as both LHS and RHS.
                # LHS: identifiers are valid determiners (key1→p1publisher).
                # RHS: identifier↔identifier FDs exist in data (key1↔p1id).
                # Empirical scoring naturally filters false positives: high-cardinality
                # RHS columns have near-zero empirical accuracy for non-determining LHS.
                return True, True
            return False, False
        if role in {"identifier", "quasi_identifier"}:
            return True, False

        strict_lhs, strict_rhs = self._policy_searchable_flags(policy)
        mode = self._search_space_mode()
        if mode == "strict":
            return strict_lhs, strict_rhs

        searchable_lhs = model_type in {"categorical", "discrete_numeric", "continuous_bucket"}
        if mode == "permissive":
            searchable_rhs = model_type in {"categorical", "discrete_numeric", "continuous_bucket"}
        else:
            searchable_rhs = model_type in {"categorical", "discrete_numeric"}

        searchable_lhs = strict_lhs or searchable_lhs
        searchable_rhs = strict_rhs or searchable_rhs
        return searchable_lhs, searchable_rhs

    def _searchability_profile(
        self,
        runtime: DatasetRuntime,
        column_name: str,
        policy,
        model_type: str,
        searchable_lhs: bool,
        searchable_rhs: bool,
    ) -> dict[str, object]:
        strict_lhs, strict_rhs = self._policy_searchable_flags(policy)
        return {
            "search_space_mode": self._search_space_mode(),
            "model_type": model_type,
            "is_target": bool(runtime.is_target(column_name)),
            "constant_like": bool(self._is_constant_like(policy)),
            "policy_searchable_lhs": bool(strict_lhs),
            "policy_searchable_rhs": bool(strict_rhs),
            "runtime_searchable_lhs": bool(searchable_lhs),
            "runtime_searchable_rhs": bool(searchable_rhs),
            "lhs_source": (
                "policy" if strict_lhs else ("relaxed" if searchable_lhs else "disabled")
            ),
            "rhs_source": (
                "policy" if strict_rhs else ("relaxed" if searchable_rhs else "disabled")
            ),
        }

    def _fit_categorical_schema(
        self,
        runtime: DatasetRuntime,
        column_name: str,
        train_values: np.ndarray,
        searchable_lhs: bool,
        searchable_rhs: bool,
        policy,
    ) -> ColumnSchema:
        normalized = [
            self._categorical_token_from_raw(runtime, column_name, value)
            for value in train_values.tolist()
        ]
        vocab, counts = self._build_vocab(normalized, self.data_config.rare_token_min_freq)
        return ColumnSchema(
            name=column_name,
            column_type="categorical",
            searchable_lhs=searchable_lhs,
            searchable_rhs=searchable_rhs,
            vocab=vocab,
            stats={
                "train_unique": len(set(normalized)),
                "raw_train_unique": int(np.unique(train_values).size),
                "top_count": int(max(counts.values()) if counts else 0),
                "rare_token_enabled": bool(self.data_config.rare_token_min_freq > 1),
                "policy": policy.to_profile(),
            },
        )

    def _fit_discrete_numeric_schema(
        self,
        column_name: str,
        train_values: np.ndarray,
        searchable_lhs: bool,
        searchable_rhs: bool,
        policy,
    ) -> ColumnSchema:
        normalized = [canonicalize_discrete_numeric(value) for value in train_values.tolist()]
        vocab, counts = self._build_vocab(normalized, self.data_config.rare_token_min_freq)
        return ColumnSchema(
            name=column_name,
            column_type="discrete_numeric",
            searchable_lhs=searchable_lhs,
            searchable_rhs=searchable_rhs,
            vocab=vocab,
            stats={
                "train_unique": len(set(normalized)),
                "raw_train_unique": int(np.unique(train_values).size),
                "top_count": int(max(counts.values()) if counts else 0),
                "rare_token_enabled": bool(self.data_config.rare_token_min_freq > 1),
                "policy": policy.to_profile(),
            },
        )

    def _fit_continuous_bucket_schema(
        self,
        column_name: str,
        train_values: np.ndarray,
        searchable_lhs: bool,
        searchable_rhs: bool,
        policy,
    ) -> ColumnSchema:
        parsed = np.array(
            [value for value in (parse_continuous(item) for item in train_values.tolist()) if value is not None],
            dtype=np.float64,
        )
        if parsed.size == 0:
            bucket_edges = np.array([0.0, 1.0], dtype=np.float64)
        else:
            quantiles = np.linspace(0.0, 1.0, self.data_config.continuous_bins + 1)
            bucket_edges = np.quantile(parsed, quantiles).astype(np.float64)
            bucket_edges = np.unique(bucket_edges)
            if bucket_edges.size <= 1:
                bucket_edges = np.array([float(parsed.min()), float(parsed.max()) + 1.0], dtype=np.float64)

        num_bins = max(1, int(bucket_edges.size - 1))
        bucket_tokens = [f"bin_{bucket_id:02d}" for bucket_id in range(num_bins)]
        vocab = list(SPECIAL_TOKENS) + bucket_tokens
        return ColumnSchema(
            name=column_name,
            column_type="continuous_bucket",
            searchable_lhs=searchable_lhs,
            searchable_rhs=searchable_rhs,
            vocab=vocab,
            bucket_edges=bucket_edges.tolist(),
            stats={
                "train_min": float(parsed.min()) if parsed.size else None,
                "train_max": float(parsed.max()) if parsed.size else None,
                "num_bins": int(num_bins),
                "policy": policy.to_profile(),
            },
        )

    def fit_column_schema(
        self,
        runtime: DatasetRuntime,
        col_id: int,
        train_values: np.ndarray,
    ) -> ColumnSchema:
        column_name = runtime.columns[col_id]
        policy = runtime.column_policies[col_id]
        model_type = self._column_model_type(runtime, column_name, policy)
        searchable_lhs, searchable_rhs = self._searchable_flags(runtime, column_name, policy, model_type)

        if model_type == "categorical":
            column_schema = self._fit_categorical_schema(
                runtime=runtime,
                column_name=column_name,
                train_values=train_values,
                searchable_lhs=searchable_lhs,
                searchable_rhs=searchable_rhs,
                policy=policy,
            )
        elif model_type == "continuous_bucket":
            column_schema = self._fit_continuous_bucket_schema(
                column_name=column_name,
                train_values=train_values,
                searchable_lhs=searchable_lhs,
                searchable_rhs=searchable_rhs,
                policy=policy,
            )
        else:
            column_schema = self._fit_discrete_numeric_schema(
                column_name=column_name,
                train_values=train_values,
                searchable_lhs=searchable_lhs,
                searchable_rhs=searchable_rhs,
                policy=policy,
            )

        column_schema.stats["searchability"] = self._searchability_profile(
            runtime=runtime,
            column_name=column_name,
            policy=policy,
            model_type=model_type,
            searchable_lhs=searchable_lhs,
            searchable_rhs=searchable_rhs,
        )
        return column_schema

    def fit_schema(self, runtime: DatasetRuntime, splits: SplitArrays) -> DatasetSchema:
        columns = [
            self.fit_column_schema(runtime, col_id, splits.train[:, col_id])
            for col_id in range(splits.train.shape[1])
        ]
        return DatasetSchema(
            dataset_name=runtime.dataset_name,
            columns=columns,
            special_tokens=list(SPECIAL_TOKENS),
            train_rows=int(splits.train.shape[0]),
            val_rows=int(splits.val.shape[0]),
            test_rows=int(splits.test.shape[0]),
            seed=self.data_config.seed,
        )

    def encode_value(
        self,
        runtime: DatasetRuntime,
        column_schema: ColumnSchema,
        raw_value: object,
    ) -> int:
        column_name = column_schema.name

        if column_schema.column_type == "categorical":
            token = self._categorical_token_from_raw(runtime, column_name, raw_value)
            if token in column_schema.value_to_id:
                return column_schema.encode_token(token)
            if column_schema.stats.get("rare_token_enabled"):
                return column_schema.rare_id
            return column_schema.unk_id

        if column_schema.column_type == "discrete_numeric":
            token = canonicalize_discrete_numeric(raw_value)
            if token in column_schema.value_to_id:
                return column_schema.encode_token(token)
            if column_schema.stats.get("rare_token_enabled"):
                return column_schema.rare_id
            return column_schema.unk_id

        if column_schema.column_type == "continuous_bucket":
            parsed = parse_continuous(raw_value)
            if parsed is None:
                return column_schema.null_id
            edges = np.asarray(column_schema.bucket_edges, dtype=np.float64)
            bucket_id = int(np.searchsorted(edges[1:-1], parsed, side="right"))
            return column_schema.encode_token(f"bin_{bucket_id:02d}")

        raise ValueError(f"Unknown column type: {column_schema.column_type}")

    def encode_array(
        self,
        runtime: DatasetRuntime,
        array: np.ndarray,
        schema: DatasetSchema,
    ) -> np.ndarray:
        token_array = np.zeros((array.shape[0], schema.num_columns), dtype=np.int32)
        for col_id, column_schema in enumerate(schema.columns):
            token_array[:, col_id] = [
                self.encode_value(runtime, column_schema, raw_value)
                for raw_value in array[:, col_id].tolist()
            ]
        return token_array

    def run(self, force: bool = False) -> DatasetSchema:
        self.config.paths.ensure_dirs()

        if (
            not force
            and self.config.paths.schema_path.exists()
            and self.config.paths.train_tokens_path.exists()
            and self.config.paths.val_tokens_path.exists()
            and self.config.paths.test_tokens_path.exists()
            and self._existing_artifacts_match_config()
        ):
            return DatasetSchema.load(self.config.paths.schema_path)

        runtime = self.load_runtime()
        splits = self.split_array(runtime.selected_array)
        schema = self.fit_schema(runtime, splits)

        train_tokens = self.encode_array(runtime, splits.train, schema)
        val_tokens = self.encode_array(runtime, splits.val, schema)
        test_tokens = self.encode_array(runtime, splits.test, schema)

        np.save(self.config.paths.train_tokens_path, train_tokens)
        np.save(self.config.paths.val_tokens_path, val_tokens)
        np.save(self.config.paths.test_tokens_path, test_tokens)
        schema.save(self.config.paths.schema_path)

        summary = {
            "preprocess_version": PREPROCESS_VERSION,
            "dataset_name": runtime.dataset_name,
            "rows_total": int(runtime.selected_array.shape[0]),
            "rows_train": int(splits.train.shape[0]),
            "rows_val": int(splits.val.shape[0]),
            "rows_test": int(splits.test.shape[0]),
            "columns": schema.column_names,
            "search_space_mode": self._search_space_mode(),
            "continuous_bins": self.data_config.continuous_bins,
            "rare_token_min_freq": self.data_config.rare_token_min_freq,
            "sample_rows": self.data_config.sample_rows,
            "column_policies": {
                runtime.columns[col_id]: runtime.column_policies[col_id].to_profile()
                for col_id in range(len(runtime.columns))
            },
            "column_searchability": {
                column.name: dict(column.stats.get("searchability", {}))
                for column in schema.columns
            },
            "searchable_lhs": schema.searchable_lhs_indices(),
            "searchable_rhs": schema.searchable_rhs_indices(),
        }
        with self.config.paths.preprocess_summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=True, indent=2)
        return schema


def run_preprocessing(config: PipelineConfig | None = None, force: bool = False) -> DatasetSchema:
    pipeline_config = config or default_config()
    preprocessor = DatasetPreprocessor(pipeline_config)
    return preprocessor.run(force=force)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build AR preprocessing artifacts for a supported dataset.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME, choices=SUPPORTED_DATASETS, help="Dataset name.")
    parser.add_argument("--force", action="store_true", help="Rebuild schema and token arrays.")
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
    schema = run_preprocessing(config, force=args.force)
    print(json.dumps(schema.to_dict(), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
