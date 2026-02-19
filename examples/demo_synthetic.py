#!/usr/bin/env python3
"""
Toy demo: Shapley Filtering vs Shapley Weighting vs Vanilla.

This demo intentionally uses a public/synthetic dataset so anyone can run it
without credentialed-access datasets.
"""
from __future__ import annotations

import argparse
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score

from shapley_weighting.shapley import ShapleyConfig, compute_negative_shapley
from shapley_weighting.weighting import WeightingConfig, shapley_rank_weights
from shapley_weighting.models import make_lr


def metrics(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    f1 = f1_score(y_true, (y_prob >= 0.5).astype(int))
    return {"auc": float(auc), "pr": float(pr), "brier": float(brier), "f1@0.5": float(f1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utility", choices=["roc", "pr"], default="roc")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--mc", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y = make_classification(
        n_samples=8000,
        n_features=100,
        n_informative=20,
        n_redundant=10,
        weights=[0.9, 0.1],
        flip_y=0.02,
        random_state=args.seed,
    )
    X = X.astype(np.float32)
    y = y.astype(int)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)

    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]

        # inner split for phi
        X_tr_in, X_val, y_tr_in, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=args.seed
        )

        # phi on inner split (negatives only)
        phi = compute_negative_shapley(
            X_tr_in, y_tr_in, X_val, y_val,
            ShapleyConfig(mc_permutations=args.mc, utility=args.utility, seed=args.seed),
        )

        # --- Vanilla ---
        m = make_lr(seed=args.seed)
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        rows.append(("Vanilla", fold, metrics(y_te, p)))

        # --- Shapley Filtering baseline (remove negatives with phi < 0, only within inner-train index space) ---
        # For a clean demo we apply filtering to the inner-train portion only, then refit on that subset.
        keep = np.ones(len(y_tr_in), dtype=bool)
        for i, v in phi.items():
            if v < 0.0:
                keep[int(i)] = False
        # build filtered training by combining kept inner-train + full positives from inner-train (already included) + remaining outer-train not in inner-train
        # (Simplified: just train on kept inner-train + all validation split)
        X_f = np.concatenate([X_tr_in[keep], X_val], axis=0)
        y_f = np.concatenate([y_tr_in[keep], y_val], axis=0)
        if len(np.unique(y_f)) == 2:
            mf = make_lr(seed=args.seed)
            mf.fit(X_f, y_f)
            pf = mf.predict_proba(X_te)[:, 1]
            rows.append(("SF (demo)", fold, metrics(y_te, pf)))

        # --- Shapley Weighting (rank-based) on inner-train indices, then train weighted on full outer-train ---
        # We compute weights for inner-train indices only and embed them back to the full outer-train as 1s elsewhere (simplified).
        w_inner, stats = shapley_rank_weights(
            y_tr_in, phi, WeightingConfig(gamma=args.gamma)
        )
        # Build weights on full outer-train: default 1, override for the inner-train portion (first part only in this simplified demo)
        w_full = np.ones(len(y_tr), dtype=np.float32)
        # To keep the demo simple, we apply weights to the first len(inner) samples after we reconstruct a consistent array.
        # Here we just train on (inner-train + val) with weights for inner-train and 1 for val.
        X_w = np.concatenate([X_tr_in, X_val], axis=0)
        y_w = np.concatenate([y_tr_in, y_val], axis=0)
        w_w = np.concatenate([w_inner, np.ones(len(y_val), dtype=np.float32)], axis=0)

        mw = make_lr(seed=args.seed)
        mw.fit(X_w, y_w, lr__sample_weight=w_w)  # pipeline: pass to final step
        pw = mw.predict_proba(X_te)[:, 1]
        out = metrics(y_te, pw)
        out.update(stats)
        rows.append(("SW (demo)", fold, out))

    # print summary
    def agg(method):
        ms = [r[2] for r in rows if r[0] == method]
        keys = sorted(ms[0].keys())
        out = {}
        for k in keys:
            vals = [m[k] for m in ms if k in m]
            out[k] = (float(np.mean(vals)), float(np.std(vals)))
        return out

    methods = sorted(set(r[0] for r in rows))
    print("\n=== 3-fold summary (mean ± std) ===")
    for m in methods:
        s = agg(m)
        printable = ", ".join([f"{k}={s[k][0]:.4f}±{s[k][1]:.4f}" for k in s.keys() if k in ["auc","pr","brier","f1@0.5","ess_total","ess_neg"]])
        print(f"{m}: {printable}")


if __name__ == "__main__":
    main()
