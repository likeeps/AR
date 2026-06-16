from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from AR.column_signals import column_identifier_signal, estimate_pattern_count, inverse_ramp, ramp
from AR.datasets import DatasetRuntime, load_runtime_dataset

if TYPE_CHECKING:
    from AR.config import PipelineConfig


PROFILE_SAMPLE_ROWS = 20000


@dataclass(frozen=True)
class AutoProfileSummary:
    # 抽样后实际参与画像分析的行数。
    rows: int
    # 可作为 FD 左侧候选列的数量。
    searchable_lhs_count: int
    # 可作为 FD 右侧候选列的数量。
    searchable_rhs_count: int
    # 左侧列中“像标识符”的整体强度。
    identifier_lhs_strength: float
    # 右侧低基数列的整体强度。
    low_card_rhs_strength: float
    # 右侧高基数列的整体强度。
    high_card_rhs_strength: float
    # 近似唯一列模式数的中位数。
    identifier_pattern_median: float
    # 近似唯一列模式数的 75 分位数。
    identifier_pattern_p75: float


def _clip(value: float, lower: float, upper: float) -> float:
    # 统一转成 Python float，避免配置对象里混入 numpy 标量。
    return float(np.clip(value, lower, upper))


def _ramp(value: float, start: float, end: float) -> float:
    return ramp(value, start, end)


def _inverse_ramp(value: float, start: float, end: float) -> float:
    return inverse_ramp(value, start, end)


def _is_constant_like(policy) -> bool:
    # 常量列或唯一值数不超过 1 的列，对搜索基本没有信息量。
    return str(getattr(policy, "role", "")) == "constant" or int(getattr(policy, "unique_count", 0)) <= 1


def _column_model_type(runtime: DatasetRuntime, column_name: str, policy) -> str:
    # 将列归一到搜索阶段关心的几种建模形态。
    if runtime.is_categorical(column_name) or policy.family == "categorical":
        return "categorical"
    if runtime.is_continuous(column_name) and policy.analysis_mode != "exact":
        return "continuous_bucket"
    if policy.analysis_mode == "binned":
        return "continuous_bucket"
    return "discrete_numeric"


def _searchable_flags(runtime: DatasetRuntime, column_name: str, policy, search_space_mode: str) -> tuple[bool, bool]:
    # 目标列和常量列不进入 FD 搜索空间。
    if runtime.is_target(column_name) or _is_constant_like(policy):
        return False, False

    role = str(getattr(policy, "role", ""))
    if role == "near_identifier":
        if search_space_mode == "permissive":
            # 宽松模式下允许近标识符列同时出现在左右两侧，
            # 因为数据中可能存在接近双向的标识符关系。
            return True, True
        return False, False
    if role in {"identifier", "quasi_identifier"}:
        # 标识符类列适合作为决定项，但通常不适合作为被决定项。
        return True, False

    strict_lhs = policy.analysis_mode in {"exact", "binned"}
    strict_rhs = policy.rhs_mode == "exact"
    if search_space_mode == "strict":
        # 严格模式只接受策略明确允许的精确/分箱搜索。
        return strict_lhs, strict_rhs

    model_type = _column_model_type(runtime, column_name, policy)
    searchable_lhs = model_type in {"categorical", "discrete_numeric", "continuous_bucket"}
    if search_space_mode == "permissive":
        searchable_rhs = model_type in {"categorical", "discrete_numeric", "continuous_bucket"}
    else:
        searchable_rhs = model_type in {"categorical", "discrete_numeric"}

    return strict_lhs or searchable_lhs, strict_rhs or searchable_rhs


def _column_unique_ratio(runtime: DatasetRuntime, col_id: int) -> float:
    # 从列统计里读取唯一值比例，缺失时安全回退到 0。
    stats = runtime.column_stats.get(int(col_id), {})
    return float(stats.get("unique_ratio", 0.0) or 0.0)


def _column_identifier_signal(runtime: DatasetRuntime, col_id: int, lhs_identifier_start_ratio: float, lhs_identifier_full_ratio: float) -> float:
    policy = runtime.column_policies[int(col_id)]
    return column_identifier_signal(
        role=str(getattr(policy, "role", "")),
        unique_ratio=_column_unique_ratio(runtime, col_id),
        empirical_weight=float(getattr(policy, "empirical_weight", 0.0) or 0.0),
        lhs_identifier_start_ratio=lhs_identifier_start_ratio,
        lhs_identifier_full_ratio=lhs_identifier_full_ratio,
    )


def _estimated_pattern_count(runtime: DatasetRuntime, col_id: int) -> float:
    # 根据抽样观测估算该列可能存在的离散模式规模。
    policy = runtime.column_policies[int(col_id)]
    selected_rows = int(runtime.selected_array.shape[0])
    total_rows = int(runtime.source_total_rows) if runtime.source_total_rows > 0 else selected_rows
    return estimate_pattern_count(
        observed_unique_count=float(getattr(policy, "unique_count", 0.0) or 0.0),
        observed_unique_ratio=_column_unique_ratio(runtime, col_id),
        observed_rows=selected_rows,
        total_rows=total_rows,
    )


def _identifier_pattern_gate(lhs_signal: float, estimated_patterns: float) -> bool:
    # 同时满足“像标识符”与“模式足够多”时，才视为真正稀疏的近唯一列。
    if lhs_signal < 0.60:
        return False
    return estimated_patterns >= 128.0


def _build_profile_summary(runtime: DatasetRuntime, search_space_mode: str, lhs_identifier_start_ratio: float, lhs_identifier_full_ratio: float) -> AutoProfileSummary:
    # 汇总列级信号，得到后续自动调参所需的整体画像。
    lhs_ids: list[int] = []
    rhs_ids: list[int] = []
    identifier_signals: list[float] = []
    rhs_low_card_signals: list[float] = []
    rhs_high_card_signals: list[float] = []
    identifier_pattern_counts: list[float] = []

    for col_id, column_name in enumerate(runtime.columns):
        policy = runtime.column_policies[col_id]
        searchable_lhs, searchable_rhs = _searchable_flags(runtime, column_name, policy, search_space_mode)
        unique_ratio = _column_unique_ratio(runtime, col_id)

        if searchable_lhs:
            lhs_ids.append(col_id)
            # LHS 侧主要关心近标识符强度，因为它会影响搜索噪声和方向判断。
            lhs_signal = _column_identifier_signal(
                runtime,
                col_id,
                lhs_identifier_start_ratio=lhs_identifier_start_ratio,
                lhs_identifier_full_ratio=lhs_identifier_full_ratio,
            )
            identifier_signals.append(lhs_signal)
            estimated_patterns = _estimated_pattern_count(runtime, col_id)
            # 只统计真正离散/分类型的近唯一列，避免连续列分箱后误抬高稀疏度信号。
            is_binned = str(getattr(policy, "analysis_mode", "")) == "binned"
            if not is_binned and _identifier_pattern_gate(lhs_signal, estimated_patterns):
                identifier_pattern_counts.append(estimated_patterns)

        if searchable_rhs:
            rhs_ids.append(col_id)
            role = str(getattr(policy, "role", ""))
            # 排除标识符类列，避免它们因为近唯一而虚增 RHS 高基数信号。
            if role not in {"identifier", "quasi_identifier", "near_identifier"} and unique_ratio < 0.30:
                rhs_low_card_signals.append(_inverse_ramp(unique_ratio, 0.02, 0.12))
                rhs_high_card_signals.append(np.sqrt(_ramp(unique_ratio, 0.02, 0.20)))
            # 中等基数列也会增加学习难度，因此记入部分高基数信号。
            elif role not in {"identifier", "quasi_identifier", "near_identifier"} and unique_ratio < 0.60:
                rhs_high_card_signals.append(0.5 * np.sqrt(_ramp(unique_ratio, 0.20, 0.60)))

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def _percentile(values: list[float], q: float) -> float:
        return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0

    def _strength(values: list[float]) -> float:
        if not values:
            return 0.0
        return max(_mean(values), _percentile(values, 75.0))

    return AutoProfileSummary(
        rows=int(runtime.selected_array.shape[0]),
        searchable_lhs_count=len(lhs_ids),
        searchable_rhs_count=len(rhs_ids),
        identifier_lhs_strength=_strength(identifier_signals),
        low_card_rhs_strength=_strength(rhs_low_card_signals),
        high_card_rhs_strength=_strength(rhs_high_card_signals),
        identifier_pattern_median=_percentile(identifier_pattern_counts, 50.0),
        identifier_pattern_p75=_percentile(identifier_pattern_counts, 75.0),
    )


@lru_cache(maxsize=32)
def _cached_profile(
    repo_root: str,
    dataset_name: str,
    search_space_mode: str,
    lhs_identifier_start_ratio: float,
    lhs_identifier_full_ratio: float,
    sample_rows: int | None,
) -> AutoProfileSummary:
    # 相同数据集与相同画像参数直接复用结果，减少重复加载和统计。
    runtime = load_runtime_dataset(
        dataset_name,
        repo_root,
        sample_rows=sample_rows,
    )
    return _build_profile_summary(
        runtime,
        search_space_mode=search_space_mode,
        lhs_identifier_start_ratio=lhs_identifier_start_ratio,
        lhs_identifier_full_ratio=lhs_identifier_full_ratio,
    )


def profile_summary(config: "PipelineConfig") -> AutoProfileSummary:
    # 自动画像最多抽样固定行数，在速度和稳定性之间取平衡。
    requested_rows = config.data.sample_rows
    profile_rows = PROFILE_SAMPLE_ROWS if requested_rows is None else min(int(requested_rows), PROFILE_SAMPLE_ROWS)
    if profile_rows <= 0:
        profile_rows = None
    return _cached_profile(
        repo_root=str(Path(config.paths.repo_root)),
        dataset_name=config.paths.dataset_name,
        search_space_mode=str(config.search.search_space_mode).strip().lower(),
        lhs_identifier_start_ratio=float(config.search.lhs_identifier_start_ratio),
        lhs_identifier_full_ratio=float(config.search.lhs_identifier_full_ratio),
        sample_rows=profile_rows,
    )


def apply_auto_profile(config: "PipelineConfig") -> "PipelineConfig":
    # 根据数据画像动态调节训练和搜索相关超参数。
    profile = profile_summary(config)

    id_lowcard_need = profile.identifier_lhs_strength * profile.low_card_rhs_strength
    empirical_need = max(profile.high_card_rhs_strength, id_lowcard_need)

    # AFD 监督始终保留；高基数越强，训练权重和训练时长越需要上调。
    config.training.afd_loss_weight = _clip(0.25 + 0.20 * empirical_need, 0.25, 0.45)
    config.training.max_epochs = int(np.clip(10 + round(10.0 * empirical_need), 10, 24))
    config.training.early_stop_patience = int(np.clip(3 + round(3.0 * empirical_need), 3, 6))

    # RHS 越偏高基数，支持集阈值就越需要放宽，避免有效样本不足。
    config.search.support_beta = _clip(12.0 + 36.0 * profile.high_card_rhs_strength, 12.0, 48.0)
    config.search.min_support_count = 2 if profile.high_card_rhs_strength >= 0.45 else 3
    config.search.max_support_rows = int(np.clip(1024 + round(1024.0 * profile.high_card_rhs_strength), 1024, 2048))
    config.search.support_head_rows = int(np.clip(config.search.max_support_rows // 2, 256, 1024))
    config.search.min_retained_mass = _clip(0.02 - 0.01 * profile.high_card_rhs_strength, 0.01, 0.02)
    config.search.min_effective_retained_mass = _clip(0.01 * profile.high_card_rhs_strength, 0.0, 0.01)

    # RHS 基数分布越复杂，模型分数通常越低，因此分数阈值需要自适应放宽。
    cardinality_diversity = max(profile.high_card_rhs_strength, profile.low_card_rhs_strength * 0.5)
    base_threshold = _clip(0.88 - 0.18 * cardinality_diversity, 0.70, 0.88)
    config.search.min_s_ent = base_threshold
    config.search.min_s_acc = base_threshold
    config.search.min_score = base_threshold

    config.search.model_score_weight = _clip(0.80 - 0.45 * empirical_need, 0.35, 0.80)
    config.search.empirical_aux_weight = _clip(0.05 + 0.10 * id_lowcard_need, 0.05, 0.15)
    config.search.empirical_high_card_base = _clip(0.10 + 0.30 * profile.high_card_rhs_strength, 0.10, 0.45)
    config.search.empirical_high_card_bonus = _clip(0.15 + 0.25 * profile.high_card_rhs_strength, 0.15, 0.40)

    config.search.coverage_penalty_weight = _clip(0.25 - 0.10 * empirical_need, 0.10, 0.25)
    config.search.delta_gain = _clip(0.04 + 0.06 * profile.high_card_rhs_strength, 0.04, 0.10)
    # 左侧越像标识符、右侧越高基数，方向判定就越需要留出安全边际。
    id_signal = max(profile.identifier_lhs_strength, profile.high_card_rhs_strength * 0.3)
    config.search.min_direction_margin = _clip(-0.02 + 0.10 * id_signal, -0.02, 0.08)

    # 仅在确实存在近唯一且模式很多的 LHS 列时，提高非空覆盖率要求。
    pattern_sparsity = _clip(profile.identifier_pattern_p75 / 5000.0, 0.0, 1.0)
    col_density = _clip(profile.searchable_lhs_count / 15.0, 0.0, 1.0)
    sparsity_signal = pattern_sparsity * col_density
    config.search.min_weighted_non_null_ratio = _clip(0.25 * sparsity_signal, 0.0, 0.25)

    # 列更多且 RHS 更复杂时，允许尝试更大的 LHS 组合。
    if profile.searchable_lhs_count >= 8 and profile.high_card_rhs_strength >= 0.3:
        config.search.max_lhs_size = 3
    else:
        config.search.max_lhs_size = 2

    return config
