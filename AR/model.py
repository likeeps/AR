from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from AR.config import ModelConfig
from AR.schema import DatasetSchema


@dataclass
class ModelMetadata:
    """Compact hyperparameter snapshot used by logs and checkpoints."""

    d_model: int
    n_heads: int
    n_layers: int
    ffn_dim: int
    dropout: float


class AnyOrderConditionalTransformer(nn.Module):
    """Transformer over table columns for any-order conditional prediction.

    Each row is treated as a short sequence whose positions correspond to columns.
    For a given batch item, some columns are marked as observed evidence and the
    others may be masked. The encoder produces one contextual representation per
    column, and each column has its own classification head because every column
    has its own vocabulary size.
    """

    def __init__(self, schema: DatasetSchema, config: ModelConfig) -> None:
        super().__init__()
        self.schema = schema
        self.config = config
        self.num_columns = schema.num_columns

        # Column embedding tells the model which logical attribute each position
        # represents. Unlike standard language modeling, position 0 is not "the
        # first token in a sentence" but "the first column in the schema".
        self.column_embedding = nn.Embedding(self.num_columns, config.d_model)

        # A 2-way embedding indicates whether a column value is currently given
        # as evidence (observed) or hidden from the model.
        self.observed_embedding = nn.Embedding(2, config.d_model)

        # Type ids are defined globally and may be sparse for a specific dataset
        # (for example, a dataset can use type ids {0, 2} without ever using 1).
        # Type embedding lets the model distinguish, for example, categorical
        # columns from bucketized continuous columns.
        self.type_embedding = nn.Embedding(max(schema.type_ids) + 1, config.d_model)

        # Each column owns an independent token embedding table because token ids
        # are local to that column's vocabulary and cannot be shared safely.
        self.value_embeddings = nn.ModuleList(
            nn.Embedding(column.vocab_size, config.d_model)
            for column in schema.columns
        )

        # The encoder is shared across all columns. Attention allows information
        # from the observed columns to flow into the hidden state of masked ones.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.output_norm = nn.LayerNorm(config.d_model)

        # Each output head maps the contextual representation of one column back
        # to that column's own discrete vocabulary.
        self.output_heads = nn.ModuleList(
            nn.Linear(config.d_model, column.vocab_size)
            for column in schema.columns
        )

        # These buffers are schema-derived constants reused in forward passes.
        # They move with the module across CPU/GPU but are not trainable.
        column_ids = torch.arange(self.num_columns, dtype=torch.long)
        type_ids = torch.tensor(schema.type_ids, dtype=torch.long)
        mask_token_ids = torch.tensor(schema.mask_token_ids, dtype=torch.long)
        self.register_buffer("column_ids", column_ids, persistent=False)
        self.register_buffer("type_ids", type_ids, persistent=False)
        self.register_buffer("mask_token_ids", mask_token_ids, persistent=False)

    def encode(self, token_ids: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        """Encode a batch of rows into contextual hidden states.

        Args:
            token_ids: Tensor of shape [batch_size, num_columns]. Each entry is a
                column-local token id.
            observed_mask: Boolean/binary tensor with the same shape. A value of
                1 means that column is visible evidence for that row; 0 means the
                model should treat it as unobserved/masked context.

        Returns:
            Tensor of shape [batch_size, num_columns, d_model] containing one
            contextual representation per column.
        """

        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must be rank-2, got {token_ids.shape}")
        if observed_mask.shape != token_ids.shape:
            raise ValueError(
                f"observed_mask must match token_ids shape, got {observed_mask.shape} vs {token_ids.shape}"
            )

        # Embed every column with its own vocabulary table, then stack the
        # per-column embeddings back into a table-shaped tensor.
        pieces = []
        for col_id, value_embedding in enumerate(self.value_embeddings):
            pieces.append(value_embedding(token_ids[:, col_id]))
        value_tensor = torch.stack(pieces, dim=1)

        # Broadcast column/type descriptors across the whole batch, while the
        # observed embedding is row-specific because each row can mask a
        # different subset of columns.
        column_embeddings = self.column_embedding(self.column_ids).unsqueeze(0)
        type_embeddings = self.type_embedding(self.type_ids).unsqueeze(0)
        observed_embeddings = self.observed_embedding(observed_mask.long())

        # The final input representation for each column is the sum of:
        # 1. its current value token embedding
        # 2. its column identity embedding
        # 3. its coarse column-type embedding
        # 4. whether it is observed or hidden in this training/query instance
        hidden = value_tensor + column_embeddings + type_embeddings + observed_embeddings
        hidden = self.encoder(hidden)
        hidden = self.output_norm(hidden)
        return hidden

    def column_logits(self, hidden: torch.Tensor, column_id: int) -> torch.Tensor:
        """Project one column's hidden state to logits over that column's vocab."""

        return self.output_heads[int(column_id)](hidden[:, int(column_id), :])

    def summary(self) -> ModelMetadata:
        """Return the minimal architecture description for reporting/debugging."""

        return ModelMetadata(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            ffn_dim=self.config.ffn_dim,
            dropout=self.config.dropout,
        )
