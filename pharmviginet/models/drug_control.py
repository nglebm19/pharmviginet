"""
Drug-level-only control for Task A.

A gradient-boosted classifier that sees only drug-level features (report volume,
number of PTs reported, first year seen, number of ingredients) — nothing about
the event or the pair. Within one PT it can only rank drugs by how "label-rich"
they look. A method that does not beat this control is not using pair-level signal.

Scores are out-of-fold: fold k is predicted by a model trained on the other folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from pharmviginet.config import TASK_A_SEED

FEATURES = ["log_n_drug", "n_pt", "first_year", "n_ingredients"]


def oof_scores(pairs: pd.DataFrame, seed: int = TASK_A_SEED) -> np.ndarray:
    out = np.full(len(pairs), np.nan)
    X = pairs[FEATURES].to_numpy(float)
    y = pairs["label"].to_numpy(int)
    fold = pairs["fold"].to_numpy()
    for k in np.unique(fold):
        test = fold == k
        model = HistGradientBoostingClassifier(random_state=seed)
        model.fit(X[~test], y[~test])
        out[test] = model.predict_proba(X[test])[:, 1]
    return out
