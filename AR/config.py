from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from AR.runtime_settings import ACTIVE_DATASET_NAME, get_runtime_overrides


SPECIAL_TOKENS = ("[NULL]", "[UNK]", "[MASK]", "[RARE]")
DEFAULT_DATASET_NAME = ACTIVE_DATASET_NAME


def _get_supported_datasets() -> tuple[str, ...]:
    """Derive SUPPORTED_DATASETS from the registry — no manual maintenance needed."""
    from train.dataset_registry import get_supported_datasets
    return get_supported_datasets()


SUPPORTED_DATASETS = _get_supported_datasets()


def validate_dataset_name(dataset_name: str) -> str:
    normalized = str(dataset_name).strip().lower()
    if normalized not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Available: {', '.join(SUPPORTED_DATASETS)}"
        )
    return normalized


@dataclass
class PathConfig:
    repo_root: str = field(default_factory=lambda: str(Path(__file__).resolve().parents[1]))
    package_root: str = field(default_factory=lambda: str(Path(__file__).resolve().parent))
    dataset_name: str = DEFAULT_DATASET_NAME
    source_csv_path: str | None = None

    def __post_init__(self) -> None:
        self.dataset_name = validate_dataset_name(self.dataset_name)

    @property
    def repo_root_path(self) -> Path:
        return Path(self.repo_root)

    @property
    def package_root_path(self) -> Path:
        return Path(self.package_root)

    @property
    def csv_path(self) -> Path:
        if self.source_csv_path:
            candidate = Path(self.source_csv_path).expanduser()
            if candidate.is_absolute():
                return candidate
            return self.repo_root_path / candidate
        from train.dataset_registry import get_dataset_spec
        spec = get_dataset_spec(self.dataset_name)
        return self.repo_root_path / "traindata" / spec.input_csv

    @property
    def source_npy_path(self) -> Path:
        from train.dataset_registry import get_dataset_spec
        spec = get_dataset_spec(self.dataset_name)
        return self.repo_root_path / "traindata" / spec.output_npy

    @property
    def metadata_path(self) -> Path:
        from train.dataset_registry import get_dataset_spec
        spec = get_dataset_spec(self.dataset_name)
        return self.repo_root_path / "traindata" / spec.meta_json

    @property
    def artifact_root(self) -> Path:
        return self.package_root_path / "artifacts" / self.dataset_name

    @property
    def checkpoint_dir(self) -> Path:
        return self.artifact_root / "checkpoints"

    @property
    def schema_path(self) -> Path:
        return self.artifact_root / "schema.json"

    @property
    def train_tokens_path(self) -> Path:
        return self.artifact_root / "train_tokens.npy"

    @property
    def val_tokens_path(self) -> Path:
        return self.artifact_root / "val_tokens.npy"

    @property
    def test_tokens_path(self) -> Path:
        return self.artifact_root / "test_tokens.npy"

    @property
    def preprocess_summary_path(self) -> Path:
        return self.artifact_root / "preprocess_summary.json"

    @property
    def training_config_path(self) -> Path:
        return self.artifact_root / "training_config.json"

    @property
    def training_history_path(self) -> Path:
        return self.artifact_root / "training_history.json"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "best.pt"

    @property
    def last_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "last.pt"

    @property
    def temperature_path(self) -> Path:
        return self.artifact_root / "temperature_scaling.json"

    @property
    def search_results_path(self) -> Path:
        return self.artifact_root / "search_results.jsonl"

    @property
    def search_summary_path(self) -> Path:
        return self.artifact_root / "search_summary.json"

    @property
    def groundtruth_path(self) -> Path:
        return self.repo_root_path / "rule_mining" / "groundtruth" / "real_world_data" / f"{self.dataset_name}.txt"

    @property
    def discovered_report_path(self) -> Path:
        return self.repo_root_path / "rule_mining" / "discovered_fd" / f"{self.dataset_name}_conditional_discovered.txt"

    def ensure_dirs(self) -> None:
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    seed: int = 20260421
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    continuous_bins: int = 32
    rare_token_min_freq: int = 1
    sample_rows: int | None = None


@dataclass
class ModelConfig:
    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 4
    ffn_dim: int = 768
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 256
    eval_batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 15
    min_epochs: int = 2
    early_stop_patience: int = 3
    grad_clip_norm: float = 1.0
    afd_loss_weight: float = 0.25
    class_weight_gamma: float = 0.5
    mask_ratios: tuple[float, ...] = (0.2, 0.5, 0.8)
    steps_per_epoch_limit: int | None = None
    eval_steps_limit: int | None = None
    use_amp: bool = True
    device: str = "cuda"
    log_every_steps: int = 100


@dataclass
class CalibrationConfig:
    max_examples_per_rhs: int = 512
    mask_ratios: tuple[float, ...] = (0.2, 0.5, 0.8)
    optimizer_steps: int = 100
    learning_rate: float = 0.05
    ece_bins: int = 15


@dataclass
class SearchConfig:
    search_space_mode: str = "balanced"
    support_beta: float = 12.0
    min_support_count: int = 3
    min_effective_support_count: int = 1
    min_pure_support_count: int = 1
    max_support_rows: int = 1024
    support_head_rows: int = 512
    min_retained_mass: float = 0.02
    min_effective_retained_mass: float = 0.0
    min_empirical_row_purity: float = 0.98
    min_non_null_ratio: float = 0.0
    min_weighted_non_null_ratio: float = 0.0
    min_s_ent: float = 0.90
    min_s_acc: float = 0.90
    min_score: float = 0.90
    score_alpha: float = 0.60
    model_score_weight: float = 0.80
    empirical_aux_weight: float = 0.05
    empirical_high_card_base: float = 0.12
    empirical_high_card_bonus: float = 0.55
    empirical_high_card_start_ratio: float = 0.02
    empirical_high_card_full_ratio: float = 0.20
    empirical_low_card_start_ratio: float = 0.02
    empirical_low_card_full_ratio: float = 0.12
    lhs_identifier_start_ratio: float = 0.65
    lhs_identifier_full_ratio: float = 0.90
    max_empirical_blend: float = 0.90
    low_card_rhs_vocab_threshold: int = 16
    low_card_min_s_ent: float = 0.86
    low_card_min_s_acc: float = 0.88
    low_card_min_score: float = 0.90
    low_card_biased_marginal_top1: float = 0.50
    low_card_biased_threshold_bump: float = 0.03
    low_card_empirical_model_floor: float = 0.30
    low_card_empirical_gap_floor: float = 0.35
    low_card_empirical_min_effective_mass: float = 0.08
    compound_high_card_min_effective_rows: int = 16
    compound_high_card_min_effective_mass: float = 0.03
    compound_min_subset_gain: float = 0.08
    low_card_compound_min_subset_gain: float = 0.12
    low_card_holdout_min_accuracy: float = 0.90
    low_card_holdout_min_coverage: float = 0.03
    low_card_holdout_min_examples: int = 64
    low_card_compound_holdout_min_lift: float = 0.05
    support_priority_effective_alpha: float = 0.75
    support_priority_non_null_beta: float = 0.50
    support_independence_weight: float = 0.0
    high_card_lhs_unique_ratio: float = 0.01
    high_card_min_effective_support_count: int = 2
    coverage_row_target: int = 64
    coverage_effective_row_target: int = 64
    coverage_mass_target: float = 0.10
    coverage_penalty_weight: float = 0.25
    level1_top_k: int = 12
    delta_gain: float = 0.05
    min_direction_margin: float = -0.01
    max_lhs_size: int = 2
    query_batch_size: int = 256
    export_soft_contingency_top_n: int = 0
    soft_contingency_max_rows: int = 128
    lhs_allowlist: tuple[str, ...] = ()
    lhs_blocklist: tuple[str, ...] = ()
    rhs_allowlist: tuple[str, ...] = ()
    rhs_blocklist: tuple[str, ...] = ()
    group_match_mode: str = "soft"
    cross_group_penalty: float = 0.35
    # Theory fix feature flags (see theory_fix_plan.md)
    exclude_rhs_special_tokens_for_scoring: bool = True  # P2: unify RHS scoring space
    use_full_mass_expectation: bool = True                # P1: full-distribution expectation
    direction_margin_mode: str = "single_only"            # P3: "single_only" | "all" | "off"


@dataclass
class PipelineConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_dataset_overrides(config: PipelineConfig) -> PipelineConfig:
    """Apply structural overrides from the dataset spec (search_space_mode, etc.).

    All threshold-type parameters are handled by apply_auto_profile (data-driven).
    Only structural config that can't be auto-profiled goes here.
    """
    from train.dataset_registry import get_dataset_spec
    try:
        spec = get_dataset_spec(config.paths.dataset_name)
    except ValueError:
        return config
    for field_name, value in spec.structural_overrides.items():
        if hasattr(config.search, field_name):
            setattr(config.search, field_name, value)
    return config


def apply_runtime_overrides(config: PipelineConfig) -> PipelineConfig:
    runtime_overrides = get_runtime_overrides(config.paths.dataset_name)
    for field_name, value in runtime_overrides.get("training", {}).items():
        if hasattr(config.training, field_name):
            setattr(config.training, field_name, value)
    for field_name, value in runtime_overrides.get("search", {}).items():
        if hasattr(config.search, field_name):
            setattr(config.search, field_name, value)
    return config


def validate_config(config: PipelineConfig) -> PipelineConfig:
    max_lhs_size = int(config.search.max_lhs_size)
    if max_lhs_size < 1:
        raise ValueError(f"search.max_lhs_size must be >= 1, got {max_lhs_size}")
    config.search.max_lhs_size = max_lhs_size

    config.search.high_card_lhs_unique_ratio = float(max(config.search.high_card_lhs_unique_ratio, 0.0))
    config.search.high_card_min_effective_support_count = max(
        1,
        int(config.search.high_card_min_effective_support_count),
    )
    return config


def default_config(
    dataset_name: str | None = None,
    *,
    source_csv_path: str | None = None,
    force_source: bool = False,
    prepare_source: bool = True,
) -> PipelineConfig:
    from AR.auto_profile import apply_auto_profile  # local import to avoid circular
    config = PipelineConfig(
        paths=PathConfig(
            dataset_name=dataset_name or DEFAULT_DATASET_NAME,
            source_csv_path=source_csv_path,
        )
    )
    config = apply_dataset_overrides(config)  # structural overrides (search_space_mode, etc.)
    if prepare_source:
        from AR.datasets import ensure_source_dataset
        ensure_source_dataset(
            config.paths.dataset_name,
            config.paths.repo_root_path,
            force=force_source,
            source_csv_path=config.paths.source_csv_path,
        )
        config = apply_auto_profile(config)       # data-driven thresholds
        config = apply_dataset_overrides(config)  # re-apply structural overrides (auto_profile may overwrite)
    config = apply_runtime_overrides(config)  # manual overrides win over everything
    return validate_config(config)
