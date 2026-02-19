"""
Rank-based Shapley Weighting (SW).

- positives: weight = 1
- negatives: weight = (rank(phi) / N_neg)^gamma
- optionally clip and normalize weights to sum to n

This matches the paper's "rank-to-weight mapping" idea.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class WeightingConfig:
    gamma: float = 0.5
    w_min: float = 1e-6
    normalize_sum_to_n: bool = True


def shapley_rank_weights(
    y_train: np.ndarray,
    phi_neg: Dict[int, float],
    cfg: WeightingConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    y_train: labels for the *same* inner-train set indices used by phi_neg keys.
            In the toy demo we compute phi on inner-train and build weights on
            the full outer-train; here we provide a generic function operating
            on one array.

    Returns:
      weights: shape (n,)
      stats: diagnostics such as ESS
    """
    y = y_train.astype(int, copy=False)
    n = len(y)
    w = np.ones(n, dtype=np.float64)

    neg_idx = np.where(y == 0)[0]
    if len(neg_idx) == 0 or len(phi_neg) == 0:
        stats = {"ess_total": float(n), "ess_neg": float(len(neg_idx))}
        return w.astype(np.float32), stats

    # Build a Series for ranks (average rank for ties)
    # Only for indices present in phi_neg
    phi_s = pd.Series(phi_neg, dtype="float64")
    ranks = (phi_s.rank(method="average") / len(phi_s)).astype("float64")  # in (0,1]
    # Apply weights to negatives with available phi; otherwise keep 1
    for i in neg_idx:
        if i in ranks.index:
            w[i] = float(ranks.loc[i]) ** float(cfg.gamma)

    w = np.clip(w, cfg.w_min, None)
    if cfg.normalize_sum_to_n:
        w *= (n / w.sum())

    # Effective sample size (ESS)
    ess_total = (w.sum() ** 2) / (np.sum(w ** 2) + 1e-12)
    w_neg = w[neg_idx] if len(neg_idx) else np.array([0.0])
    ess_neg = (w_neg.sum() ** 2) / (np.sum(w_neg ** 2) + 1e-12) if len(neg_idx) else float("nan")

    stats = {"ess_total": float(ess_total), "ess_neg": float(ess_neg)}
    return w.astype(np.float32), stats
