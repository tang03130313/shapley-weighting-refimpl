"""
Simple model builders for the toy demo.
"""
from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def make_lr(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(max_iter=500, solver="lbfgs", n_jobs=1, random_state=seed)),
    ])
