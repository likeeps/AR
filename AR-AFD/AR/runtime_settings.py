from __future__ import annotations

from typing import Any


# Change this value to switch the default dataset used by preprocess/train/calibrate/search.
ACTIVE_DATASET_NAME = "tax"

# Set to any positive integer. 1 = singleton LHS only; 2 = up to pairs; 3+ = compound LHS.
# Larger values use CAFD-style correlation pruning to keep search tractable.
ACTIVE_MAX_LHS_SIZE = 3

# Search-space mode controls how strongly column policy can prune candidate LHS/RHS columns.
# strict: keep legacy policy gates
# balanced: relax to all tokenizable LHS plus categorical/discrete RHS
# permissive: maximize recall, including bucketed continuous RHS
ACTIVE_SEARCH_SPACE_MODE = "balanced"


# Easy-to-edit global overrides.
# Delete a key or set it to None to fall back to the dataset default in AR/config.py.
GLOBAL_TRAINING_OVERRIDES: dict[str, Any] = {
    # "afd_loss_weight": 0.30,
    # "max_epochs": 10,
}

GLOBAL_SEARCH_OVERRIDES: dict[str, Any] = {
    "max_lhs_size": ACTIVE_MAX_LHS_SIZE,
    # search_space_mode is controlled per-dataset via DatasetSpec.structural_overrides.
    # Do NOT set it globally — it would overwrite per-dataset settings like dblp10k's "permissive".
}


# Per-dataset search overrides (non-empty only).
# Most datasets use auto_profile defaults — only add entries here when the
# auto-profiled value is demonstrably wrong.
# These are MERGED with search_overrides from the registry DatasetSpec.
DATASET_SEARCH_OVERRIDES: dict[str, dict[str, Any]] = {
    # claims and dblp10k overrides are now in the registry DatasetSpec.
    # Add entries here only for ad-hoc experimentation.
}


def _clean_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in overrides.items() if value is not None}


def get_runtime_overrides(dataset_name: str) -> dict[str, dict[str, Any]]:
    dataset_key = str(dataset_name).strip().lower()

    training_overrides: dict[str, Any] = {}
    search_overrides: dict[str, Any] = {}

    # Layer 1: search_overrides from the registry DatasetSpec (if any)
    try:
        from train.dataset_registry import get_dataset_spec
        spec = get_dataset_spec(dataset_key)
        search_overrides.update(_clean_overrides(spec.search_overrides))
    except (ValueError, ImportError):
        pass

    # Layer 2: per-dataset overrides from DATASET_SEARCH_OVERRIDES (legacy, non-empty only)
    search_overrides.update(_clean_overrides(DATASET_SEARCH_OVERRIDES.get(dataset_key, {})))

    # Layer 3: global overrides (highest priority)
    training_overrides.update(_clean_overrides(GLOBAL_TRAINING_OVERRIDES))
    search_overrides.update(_clean_overrides(GLOBAL_SEARCH_OVERRIDES))

    return {
        "training": training_overrides,
        "search": search_overrides,
    }
