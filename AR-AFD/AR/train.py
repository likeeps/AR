from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from AR.config import DEFAULT_DATASET_NAME, PipelineConfig, SUPPORTED_DATASETS, default_config, validate_config
from AR.data import TokenRowDataset, load_artifacts
from AR.masking import apply_mask_tokens, sample_observed_mask
from AR.model import AnyOrderConditionalTransformer
from AR.preprocess import run_preprocessing
from AR.schema import DatasetSchema


def choose_device(device_name: str) -> torch.device:
    """根据配置选择训练设备。

    当前实现只在显式请求 cuda 且本机 CUDA 可用时使用 GPU，
    否则统一回退到 CPU，避免配置和实际环境不一致时直接报错。
    """

    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_global_seed(seed: int) -> None:
    """固定 numpy / torch 的随机种子，尽量保证训练可复现。"""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_class_weight_tensors(
    train_tokens: np.ndarray,
    schema: DatasetSchema,
    gamma: float,
) -> list[torch.Tensor]:
    """为每一列构造类别权重，缓解高频 token 压制长尾 token 的问题。

    这里的权重是“按列单独计算”的，因为每列都有自己的词表和频率分布。
    公式近似为:

        weight(token) = (1 / (count(token) + 1)) ** gamma

    再做一次列内归一化，使不同列的损失尺度更稳定。
    """

    weights: list[torch.Tensor] = []
    for col_id, column in enumerate(schema.columns):
        counts = np.bincount(train_tokens[:, col_id], minlength=column.vocab_size).astype(np.float64)
        raw = np.power(1.0 / (counts + 1.0), gamma)
        # [MASK] 只是训练过程中的占位符，不应该被当成真实类别去鼓励预测。
        if column.mask_id < raw.shape[0]:
            raw[column.mask_id] = 0.0
        normalizer = raw[raw > 0].mean() if np.any(raw > 0) else 1.0
        normalized = raw / max(normalizer, 1e-12)
        weights.append(torch.tensor(normalized, dtype=torch.float32))
    return weights


def sample_afd_objective(
    batch_tokens: torch.Tensor,
    schema: DatasetSchema,
    generator: torch.Generator,
    max_lhs_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """随机采样一个 AFD 辅助训练任务。

    目标是显式告诉模型：
    “只给你一组 LHS 列，请你去预测 RHS 列。”

    返回：
    - observed_mask: 哪些列作为条件输入
    - target_mask: 哪些列作为预测目标
    - rhs_col: 本次采样到的 RHS 列索引
    """

    # 先从允许搜索的 RHS 列中随机抽一个目标列。
    searchable_rhs = schema.searchable_rhs_indices()
    rhs_col = int(searchable_rhs[int(torch.randint(len(searchable_rhs), (1,), generator=generator).item())])

    # LHS 候选来自允许搜索的 LHS 列，但不能和 RHS 是同一列。
    lhs_pool = [col for col in schema.searchable_lhs_indices() if col != rhs_col]

    # 随机采样 LHS 规模，与 search 侧 max_lhs_size 保持一致。
    sampled_max_lhs = max(1, min(len(lhs_pool), int(max_lhs_size)))
    lhs_size = 1 if sampled_max_lhs == 1 else int(torch.randint(1, sampled_max_lhs + 1, (1,), generator=generator).item())
    lhs_order = torch.randperm(len(lhs_pool), generator=generator)[:lhs_size].tolist()
    lhs_cols = [lhs_pool[idx] for idx in lhs_order]

    # AFD 任务里只有 LHS 被设为 observed，RHS 被设为 target。
    observed_mask = torch.zeros_like(batch_tokens, dtype=torch.bool)
    observed_mask[:, lhs_cols] = True
    target_mask = torch.zeros_like(batch_tokens, dtype=torch.bool)
    target_mask[:, rhs_col] = True
    return observed_mask, target_mask, rhs_col


def compute_masked_loss(
    model: AnyOrderConditionalTransformer,
    batch_tokens: torch.Tensor,
    observed_mask: torch.Tensor,
    target_mask: torch.Tensor,
    class_weights: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """计算一批样本上的掩码重建损失。

    训练思路是：
    1. 先把未观测列替换成各列自己的 [MASK] token
    2. 让模型根据 observed 列去编码整行
    3. 仅在 target_mask 指定的位置上计算交叉熵

    这样同一个模型既能做普通随机 mask 恢复，也能做 AFD 定向预测。
    """

    # 对未观测列打上 [MASK]，observed 列保留真实 token。
    masked_inputs = apply_mask_tokens(batch_tokens, observed_mask, model.mask_token_ids)

    # 经过 Transformer 编码后，得到每一列的上下文表示。
    hidden = model.encode(masked_inputs, observed_mask)

    total_loss = torch.zeros((), device=batch_tokens.device)
    active_columns = 0
    active_targets = 0

    # target_mask 可能同时激活多列，因此逐列取出对应 logits 并累计损失。
    for col_id in torch.nonzero(target_mask.any(dim=0), as_tuple=False).flatten().tolist():
        row_mask = target_mask[:, col_id]
        if not torch.any(row_mask):
            continue
        logits = model.column_logits(hidden, col_id)[row_mask]
        targets = batch_tokens[row_mask, col_id]
        loss = F.cross_entropy(logits, targets, weight=class_weights[col_id].to(batch_tokens.device))
        total_loss = total_loss + loss
        active_columns += 1
        active_targets += int(row_mask.sum().item())

    if active_columns == 0:
        raise RuntimeError("No active target columns were sampled for loss computation.")

    # 这里按“参与监督的列数”而不是按 token 总数平均，
    # 使多列目标场景下每个目标列对总损失的贡献更均衡。
    total_loss = total_loss / active_columns
    metrics = {
        "active_columns": float(active_columns),
        "active_targets": float(active_targets),
    }
    return total_loss, metrics


@torch.no_grad()
def evaluate_model(
    model: AnyOrderConditionalTransformer,
    loader: DataLoader,
    config: PipelineConfig,
    class_weights: list[torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    """在验证集上评估当前模型的随机 mask 重建能力。"""

    model.eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.data.seed + 11)

    loss_total = 0.0
    batch_total = 0
    steps_limit = config.training.eval_steps_limit

    for step_id, batch_tokens in enumerate(loader):
        if steps_limit is not None and step_id >= steps_limit:
            break
        batch_tokens = batch_tokens.to(device=device, dtype=torch.long, non_blocking=True)

        # 验证阶段沿用训练时的随机观测比例，但不做梯度更新。
        observed_mask = sample_observed_mask(
            batch_size=batch_tokens.shape[0],
            num_columns=batch_tokens.shape[1],
            ratios=config.training.mask_ratios,
            generator=generator,
        ).to(device)
        target_mask = ~observed_mask
        loss, _ = compute_masked_loss(model, batch_tokens, observed_mask, target_mask, class_weights)
        loss_total += float(loss.item())
        batch_total += 1

    return {
        "val_loss": loss_total / max(batch_total, 1),
        "val_batches": float(batch_total),
    }


def train_model(config: PipelineConfig | None = None, force_preprocess: bool = False) -> dict[str, float]:
    """训练 AR 的 Any-Order Conditional Transformer。

    训练入口会串起以下阶段：
    1. 预处理并加载 token 化产物
    2. 构建模型、DataLoader、类别权重、优化器
    3. 每个 batch 同时计算：
       - cond_loss：普通随机 mask 条件重建损失
       - afd_loss：定向的 LHS -> RHS 辅助损失
    4. 在验证集上选择 best checkpoint
    """

    pipeline_config = validate_config(config or default_config())
    pipeline_config.paths.ensure_dirs()

    # 训练前确保 schema 与 train/val/test token 文件可用。
    run_preprocessing(pipeline_config, force=force_preprocess)
    artifacts = load_artifacts(pipeline_config, mmap_train=True)

    set_global_seed(pipeline_config.data.seed)
    device = choose_device(pipeline_config.training.device)

    # 模型结构完全由 schema 和 model config 决定。
    model = AnyOrderConditionalTransformer(artifacts.schema, pipeline_config.model).to(device)

    train_dataset = TokenRowDataset(artifacts.train_tokens)
    val_dataset = TokenRowDataset(artifacts.val_tokens)

    train_loader = DataLoader(
        train_dataset,
        batch_size=pipeline_config.training.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=pipeline_config.training.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # 类别权重来自训练集统计，只构建一次，整个训练过程复用。
    class_weights = build_class_weight_tensors(
        np.asarray(artifacts.train_tokens, dtype=np.int64),
        artifacts.schema,
        gamma=pipeline_config.training.class_weight_gamma,
    )

    # 使用 AdamW 做优化；AMP 仅在 CUDA 下启用。
    optimizer = AdamW(
        model.parameters(),
        lr=pipeline_config.training.learning_rate,
        weight_decay=pipeline_config.training.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and pipeline_config.training.use_amp))

    history: list[dict[str, float]] = []
    best_val_loss = math.inf
    best_epoch = -1
    no_improvement = 0
    global_step = 0
    start_time = time.time()

    mask_generator = torch.Generator(device="cpu")
    mask_generator.manual_seed(pipeline_config.data.seed + 7)

    # 主训练循环：epoch -> batch -> 前向/反向 -> 验证 -> 保存 checkpoint
    for epoch in range(1, pipeline_config.training.max_epochs + 1):
        model.train()
        train_loss_total = 0.0
        batch_count = 0

        for step_id, batch_tokens in enumerate(train_loader, start=1):
            if (
                pipeline_config.training.steps_per_epoch_limit is not None
                and step_id > pipeline_config.training.steps_per_epoch_limit
            ):
                break

            batch_tokens = batch_tokens.to(device=device, dtype=torch.long, non_blocking=True)

            # cond_loss：对整行做随机 observed/unobserved 划分，
            # 学习“给定部分列，恢复其余列”的一般条件建模能力。
            observed_mask = sample_observed_mask(
                batch_size=batch_tokens.shape[0],
                num_columns=batch_tokens.shape[1],
                ratios=pipeline_config.training.mask_ratios,
                generator=mask_generator,
            ).to(device)
            target_mask = ~observed_mask

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                cond_loss, _ = compute_masked_loss(model, batch_tokens, observed_mask, target_mask, class_weights)

                # afd_loss：额外采样一个定向依赖任务，
                # 强化模型在“只给 LHS，预测 RHS”场景下的能力。
                afd_observed_mask, afd_target_mask, _ = sample_afd_objective(
                    batch_tokens,
                    artifacts.schema,
                    mask_generator,
                    max_lhs_size=pipeline_config.search.max_lhs_size,
                )
                afd_observed_mask = afd_observed_mask.to(device)
                afd_target_mask = afd_target_mask.to(device)
                afd_loss, _ = compute_masked_loss(
                    model,
                    batch_tokens,
                    afd_observed_mask,
                    afd_target_mask,
                    class_weights,
                )

                # 总损失 = 通用条件建模损失 + AFD 辅助损失。
                # afd_loss_weight 越大，模型越偏向依赖发现任务本身。
                loss = cond_loss + pipeline_config.training.afd_loss_weight * afd_loss

            if scaler.is_enabled():
                # AMP 下用 GradScaler 避免半精度梯度数值不稳定。
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), pipeline_config.training.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                clip_grad_norm_(model.parameters(), pipeline_config.training.grad_clip_norm)
                optimizer.step()

            train_loss_total += float(loss.item())
            batch_count += 1
            global_step += 1

            if global_step % pipeline_config.training.log_every_steps == 0:
                elapsed = time.time() - start_time
                print(
                    f"epoch={epoch:02d} step={global_step:06d} "
                    f"train_loss={train_loss_total / max(batch_count, 1):.4f} "
                    f"elapsed={elapsed:.1f}s"
                )

        # 每个 epoch 结束后都在验证集上评估，并据此决定是否刷新 best checkpoint。
        metrics = evaluate_model(model, val_loader, pipeline_config, class_weights, device)
        train_loss = train_loss_total / max(batch_count, 1)
        epoch_record = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            **metrics,
        }
        history.append(epoch_record)
        print(
            f"[epoch {epoch:02d}] train_loss={train_loss:.4f} "
            f"val_loss={metrics['val_loss']:.4f}"
        )

        # last.pt 始终保存最近一次训练状态，方便中断后检查。
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": asdict(pipeline_config.model),
                "schema_path": str(pipeline_config.paths.schema_path),
                "epoch": epoch,
            },
            pipeline_config.paths.last_checkpoint_path,
        )

        # best.pt 只在验证损失更优时更新，后续查询和校准都使用它。
        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            best_epoch = epoch
            no_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": asdict(pipeline_config.model),
                    "schema_path": str(pipeline_config.paths.schema_path),
                    "epoch": epoch,
                },
                pipeline_config.paths.best_checkpoint_path,
            )
        else:
            no_improvement += 1

        # 达到最小训练轮数后，如果连续若干轮没有提升则提前停止。
        if epoch >= pipeline_config.training.min_epochs and no_improvement >= pipeline_config.training.early_stop_patience:
            print(
                f"Early stop after epoch={epoch}, best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f}"
            )
            break

    # 训练完成后写出完整配置和逐 epoch 历史，方便复现实验。
    with pipeline_config.paths.training_config_path.open("w", encoding="utf-8") as handle:
        json.dump(pipeline_config.to_dict(), handle, ensure_ascii=True, indent=2)
    with pipeline_config.paths.training_history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=True, indent=2)

    summary = {
        "best_val_loss": float(best_val_loss),
        "best_epoch": float(best_epoch),
        "epochs_ran": float(len(history)),
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    """命令行参数定义，主要用于快速覆盖常用训练超参数。"""

    parser = argparse.ArgumentParser(description="Train the AR any-order conditional transformer.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME, choices=SUPPORTED_DATASETS, help="Dataset name.")
    parser.add_argument("--force-preprocess", action="store_true", help="Rebuild preprocessing artifacts first.")
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional row cap for smoke tests.")
    parser.add_argument("--continuous-bins", type=int, default=None, help="Override bucket count.")
    parser.add_argument("--epochs", type=int, default=None, help="Override max training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train batch size.")
    parser.add_argument("--afd-loss-weight", type=float, default=None, help="Override afd auxiliary loss weight.")
    parser.add_argument("--max-lhs-size", type=int, default=None, help="Override maximum lhs size used by afd sampling.")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Optional per-epoch step cap.")
    return parser


def main() -> None:
    """命令行入口：读取覆盖参数后启动训练。"""

    args = build_parser().parse_args()
    config = default_config(args.dataset)
    if args.sample_rows is not None:
        config.data.sample_rows = args.sample_rows
    if args.continuous_bins is not None:
        config.data.continuous_bins = args.continuous_bins
    if args.epochs is not None:
        config.training.max_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.afd_loss_weight is not None:
        config.training.afd_loss_weight = args.afd_loss_weight
    if args.max_lhs_size is not None:
        config.search.max_lhs_size = args.max_lhs_size
    if args.steps_per_epoch is not None:
        config.training.steps_per_epoch_limit = args.steps_per_epoch
    summary = train_model(config, force_preprocess=args.force_preprocess)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
