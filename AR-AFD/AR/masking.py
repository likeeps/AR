from __future__ import annotations

import torch


def sample_observed_mask(
    batch_size: int,
    num_columns: int,
    ratios: tuple[float, ...],
    generator: torch.Generator,
) -> torch.Tensor:
    observed = torch.zeros((batch_size, num_columns), dtype=torch.bool)
    ratio_ids = torch.randint(
        low=0,
        high=len(ratios),
        size=(batch_size,),
        generator=generator,
    )
    for row_id in range(batch_size):
        ratio = float(ratios[int(ratio_ids[row_id])])
        observed_count = max(1, min(num_columns - 1, int(round(num_columns * ratio))))
        selected = torch.randperm(num_columns, generator=generator)[:observed_count]
        observed[row_id, selected] = True
    return observed


def apply_mask_tokens(batch_tokens: torch.Tensor, observed_mask: torch.Tensor, mask_token_ids: torch.Tensor) -> torch.Tensor:
    masked = batch_tokens.clone()
    expanded_mask_ids = mask_token_ids.unsqueeze(0).expand_as(masked)
    masked = torch.where(observed_mask, masked, expanded_mask_ids)
    return masked

