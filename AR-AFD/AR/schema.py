from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TYPE_TO_ID = {
    "categorical": 0,
    "discrete_numeric": 1,
    "continuous_bucket": 2,
}


@dataclass
class ColumnSchema:
    name: str
    column_type: str
    searchable_lhs: bool
    searchable_rhs: bool
    vocab: list[str]
    bucket_edges: list[float] | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    value_to_id: dict[str, int] = field(init=False, repr=False)
    null_id: int = field(init=False)
    unk_id: int = field(init=False)
    mask_id: int = field(init=False)
    rare_id: int = field(init=False)

    def __post_init__(self) -> None:
        self.value_to_id = {value: idx for idx, value in enumerate(self.vocab)}
        self.null_id = self.value_to_id["[NULL]"]
        self.unk_id = self.value_to_id["[UNK]"]
        self.mask_id = self.value_to_id["[MASK]"]
        self.rare_id = self.value_to_id["[RARE]"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def type_id(self) -> int:
        return TYPE_TO_ID[self.column_type]

    def encode_token(self, token: str) -> int:
        return self.value_to_id.get(token, self.unk_id)

    def decode_token(self, token_id: int) -> str:
        return self.vocab[int(token_id)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "column_type": self.column_type,
            "searchable_lhs": self.searchable_lhs,
            "searchable_rhs": self.searchable_rhs,
            "vocab": self.vocab,
            "bucket_edges": self.bucket_edges,
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ColumnSchema":
        return cls(
            name=str(payload["name"]),
            column_type=str(payload["column_type"]),
            searchable_lhs=bool(payload["searchable_lhs"]),
            searchable_rhs=bool(payload["searchable_rhs"]),
            vocab=list(payload["vocab"]),
            bucket_edges=list(payload["bucket_edges"]) if payload.get("bucket_edges") is not None else None,
            stats=dict(payload.get("stats", {})),
        )


@dataclass
class DatasetSchema:
    dataset_name: str
    columns: list[ColumnSchema]
    special_tokens: list[str]
    train_rows: int
    val_rows: int
    test_rows: int
    seed: int
    name_to_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.name_to_index = {column.name: idx for idx, column in enumerate(self.columns)}

    @property
    def num_columns(self) -> int:
        return len(self.columns)

    @property
    def column_names(self) -> list[str]:
        return [column.name for column in self.columns]

    @property
    def mask_token_ids(self) -> list[int]:
        return [column.mask_id for column in self.columns]

    @property
    def type_ids(self) -> list[int]:
        return [column.type_id for column in self.columns]

    def column_index(self, column: int | str) -> int:
        if isinstance(column, int):
            return column
        return self.name_to_index[column]

    def column(self, column: int | str) -> ColumnSchema:
        return self.columns[self.column_index(column)]

    def searchable_rhs_indices(self) -> list[int]:
        return [idx for idx, column in enumerate(self.columns) if column.searchable_rhs]

    def searchable_lhs_indices(self) -> list[int]:
        return [idx for idx, column in enumerate(self.columns) if column.searchable_lhs]

    def encode_value(self, column: int | str, token: str) -> int:
        return self.column(column).encode_token(token)

    def decode_value(self, column: int | str, token_id: int) -> str:
        return self.column(column).decode_token(token_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "special_tokens": self.special_tokens,
            "train_rows": self.train_rows,
            "val_rows": self.val_rows,
            "test_rows": self.test_rows,
            "seed": self.seed,
            "columns": [column.to_dict() for column in self.columns],
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=True, indent=2)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetSchema":
        return cls(
            dataset_name=str(payload["dataset_name"]),
            columns=[ColumnSchema.from_dict(item) for item in payload["columns"]],
            special_tokens=list(payload["special_tokens"]),
            train_rows=int(payload["train_rows"]),
            val_rows=int(payload["val_rows"]),
            test_rows=int(payload["test_rows"]),
            seed=int(payload["seed"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DatasetSchema":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
