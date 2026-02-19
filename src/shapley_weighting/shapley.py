"""
Shapley attribution utilities (negative-only) for reference implementation.

Core idea:
- Compute Data Shapley values φ for negative samples only on an inner split.
- Use a lightweight base model (Logistic Regression) and a Monte Carlo
  permutation estimator with optional early stopping based on inner-val utility.

This is a simplified implementation aligned with the PAKDD 2026 paper's method,
but not tied to any specific dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable, Dict
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score


def base_model_for_phi(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(max_iter=400, solver="lbfgs", n_jobs=1, random_state=seed)),
    ])


def utility_score(y_true: np.ndarray, y_prob: np.ndarray, utility: str) -> float:
    if utility == "roc":
        return float(roc_auc_score(y_true, y_prob))
    if utility == "pr":
        return float(average_precision_score(y_true, y_prob))
    raise ValueError("utility must be 'roc' or 'pr'")


@dataclass
class ShapleyConfig:
    mc_permutations: int = 2
    early_tol: float = 1e-2
    early_steps: int = 2
    neg_cap: Optional[int] = 25_000
    seed: int = 42
    utility: str = "roc"


def compute_negative_shapley(
    X_tr_in: np.ndarray,
    y_tr_in: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: ShapleyConfig,
) -> Dict[int, float]:
    """
    Returns a dict: {neg_index_in_inner_train: phi_value}.
    Indices refer to rows of the provided inner-train arrays.
    """
    y_tr_in = y_tr_in.astype(int, copy=False)
    y_val = y_val.astype(int, copy=False)

    idx_all = np.arange(len(y_tr_in))
    pos_idx = idx_all[y_tr_in == 1]
    neg_idx = idx_all[y_tr_in == 0]
    if len(neg_idx) == 0:
        return {}

    # cap negatives used for phi
    if cfg.neg_cap is not None and len(neg_idx) > cfg.neg_cap:
        rng = np.random.default_rng(cfg.seed)
        neg_idx = np.sort(rng.choice(neg_idx, size=cfg.neg_cap, replace=False))

    # full utility (for early stopping reference)
    pipe_full = base_model_for_phi(seed=cfg.seed)
    pipe_full.fit(X_tr_in, y_tr_in)
    full_prob = pipe_full.predict_proba(X_val)[:, 1]
    full_u = utility_score(y_val, full_prob, cfg.utility)

    base_u = 0.5 if cfg.utility == "roc" else float(np.mean(y_val))

    rng = np.random.default_rng(cfg.seed)
    phi_accum = {int(i): 0.0 for i in neg_idx}

    def util_from_subset(subset_idx: list[int]) -> float:
        ys = y_tr_in[subset_idx]
        # avoid single-class training
        if ys.min() == ys.max():
            return base_u
        pipe = base_model_for_phi(seed=cfg.seed)
        pipe.fit(X_tr_in[subset_idx], ys)
        prob = pipe.predict_proba(X_val)[:, 1]
        return utility_score(y_val, prob, cfg.utility)

    for _ in range(cfg.mc_permutations):
        perm = rng.permutation(neg_idx)
        cur_idx = list(map(int, pos_idx.tolist()))
        u_old = base_u
        near_full = 0
        for neg_j in perm:
            cur_idx.append(int(neg_j))
            u_new = util_from_subset(cur_idx)
            phi_accum[int(neg_j)] += (u_new - u_old)
            u_old = u_new
            near_full = near_full + 1 if abs(u_new - full_u) <= cfg.early_tol else 0
            if near_full >= cfg.early_steps:
                break

    for k in list(phi_accum.keys()):
        phi_accum[k] /= float(cfg.mc_permutations)

    return phi_accum
