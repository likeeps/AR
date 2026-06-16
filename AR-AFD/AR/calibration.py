from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from AR.config import DEFAULT_DATASET_NAME, PipelineConfig, SUPPORTED_DATASETS, default_config
from AR.data import TokenRowDataset, load_artifacts
from AR.masking import apply_mask_tokens, sample_observed_mask
from AR.metrics import multiclass_brier_score, top_label_ece
from AR.model import AnyOrderConditionalTransformer
from AR.preprocess import run_preprocessing
from AR.train import choose_device


def load_trained_model(config: PipelineConfig, device: torch.device) -> tuple[AnyOrderConditionalTransformer, Any]:
    artifacts = load_artifacts(config, mmap_train=False)
    checkpoint = torch.load(config.paths.best_checkpoint_path, map_location=device)
    model = AnyOrderConditionalTransformer(artifacts.schema, config.model).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, artifacts.schema


@torch.no_grad()
def collect_logits_for_rhs(
    model: AnyOrderConditionalTransformer,
    data_loader: DataLoader,
    rhs_col: int,
    config: PipelineConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.data.seed + 31 + rhs_col)

    logits_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    collected = 0

    for batch_tokens in data_loader:
        if collected >= config.calibration.max_examples_per_rhs:
            break

        batch_tokens = batch_tokens.to(device=device, dtype=torch.long, non_blocking=True)
        observed_mask = sample_observed_mask(
            batch_size=batch_tokens.shape[0],
            num_columns=batch_tokens.shape[1],
            ratios=config.calibration.mask_ratios,
            generator=generator,
        ).to(device)
        target_mask = ~observed_mask[:, rhs_col]
        if not torch.any(target_mask):
            continue

        masked_inputs = apply_mask_tokens(batch_tokens, observed_mask, model.mask_token_ids)
        hidden = model.encode(masked_inputs, observed_mask)
        logits = model.column_logits(hidden, rhs_col)[target_mask]
        targets = batch_tokens[target_mask, rhs_col]

        remaining = config.calibration.max_examples_per_rhs - collected
        if logits.shape[0] > remaining:
            logits = logits[:remaining]
            targets = targets[:remaining]

        logits_parts.append(logits.detach().cpu())
        target_parts.append(targets.detach().cpu())
        collected += int(logits.shape[0])

    if not logits_parts:
        raise RuntimeError(f"No logits collected for calibration column {rhs_col}")

    return torch.cat(logits_parts, dim=0), torch.cat(target_parts, dim=0)


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, steps: int, learning_rate: float) -> tuple[float, dict[str, float]]:
    logits = logits.float()
    targets = targets.long()

    log_temperature = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([log_temperature], lr=learning_rate)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature).clamp_min(1e-3)
        loss = F.cross_entropy(logits / temperature, targets)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        temperature = float(torch.exp(log_temperature).clamp_min(1e-3).item())
        probs = torch.softmax(logits / temperature, dim=1).cpu().numpy()
        targets_np = targets.cpu().numpy()
        metrics = {
            "nll": float(F.cross_entropy(logits / temperature, targets).item()),
            "brier": multiclass_brier_score(probs, targets_np),
            "ece": top_label_ece(probs, targets_np),
        }
    return temperature, metrics


def calibrate_model(config: PipelineConfig | None = None, force_preprocess: bool = False) -> dict[str, Any]:
    pipeline_config = config or default_config()
    pipeline_config.paths.ensure_dirs()
    run_preprocessing(pipeline_config, force=force_preprocess)

    device = choose_device(pipeline_config.training.device)
    model, schema = load_trained_model(pipeline_config, device)
    artifacts = load_artifacts(pipeline_config, mmap_train=False)
    val_loader = DataLoader(
        TokenRowDataset(artifacts.val_tokens),
        batch_size=pipeline_config.training.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    temperatures = {column.name: 1.0 for column in schema.columns}
    metrics_by_column: dict[str, dict[str, float]] = {}

    for rhs_col in schema.searchable_rhs_indices():
        logits, targets = collect_logits_for_rhs(
            model=model,
            data_loader=val_loader,
            rhs_col=rhs_col,
            config=pipeline_config,
            device=device,
        )
        temperature, metrics = fit_temperature(
            logits=logits,
            targets=targets,
            steps=pipeline_config.calibration.optimizer_steps,
            learning_rate=pipeline_config.calibration.learning_rate,
        )
        column_name = schema.columns[rhs_col].name
        temperatures[column_name] = temperature
        metrics_by_column[column_name] = metrics
        print(
            f"calibrated {column_name}: temperature={temperature:.4f} "
            f"nll={metrics['nll']:.4f} ece={metrics['ece']:.4f}"
        )

    payload = {
        "temperatures": temperatures,
        "metrics": metrics_by_column,
    }
    with pipeline_config.paths.temperature_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit temperature scaling for AR search.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME, choices=SUPPORTED_DATASETS, help="Dataset name.")
    parser.add_argument("--force-preprocess", action="store_true", help="Rebuild preprocessing artifacts first.")
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional row cap for smoke tests.")
    parser.add_argument("--continuous-bins", type=int, default=None, help="Override bucket count.")
    parser.add_argument("--max-examples", type=int, default=None, help="Examples per RHS for temperature fitting.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = default_config(args.dataset)
    if args.sample_rows is not None:
        config.data.sample_rows = args.sample_rows
    if args.continuous_bins is not None:
        config.data.continuous_bins = args.continuous_bins
    if args.max_examples is not None:
        config.calibration.max_examples_per_rhs = args.max_examples
    payload = calibrate_model(config, force_preprocess=args.force_preprocess)
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
