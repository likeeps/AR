from __future__ import annotations

import numpy as np


EPS = 1e-12


def entropy_from_probs(probabilities: np.ndarray) -> np.ndarray:
    probs = np.clip(probabilities, EPS, 1.0)
    return -np.sum(probs * np.log(probs), axis=-1)


def multiclass_brier_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(targets.shape[0]), targets] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def top_label_ece(probabilities: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accuracy = (predictions == targets).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = float(targets.shape[0])
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        if right == 1.0:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)
        if not np.any(mask):
            continue
        bucket_conf = float(confidences[mask].mean())
        bucket_acc = float(accuracy[mask].mean())
        ece += float(mask.mean()) * abs(bucket_acc - bucket_conf)
    return float(ece)

