import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def stratified_auc(y_true, y_score, groups, min_n: int = 20) -> tuple[float, int]:
    """
    AUC computed within each group (e.g. each MedDRA PT), averaged weighted by
    group size. Groups with < min_n rows or a single class are skipped.
    Returns (weighted mean AUC, number of groups scored).
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "s": np.asarray(y_score), "g": np.asarray(groups)})
    aucs, weights = [], []
    for _, d in df.groupby("g", sort=False):
        if len(d) < min_n or d["y"].nunique() < 2:
            continue
        aucs.append(roc_auc_score(d["y"], d["s"]))
        weights.append(len(d))
    if not aucs:
        return float("nan"), 0
    return float(np.average(aucs, weights=weights)), len(aucs)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5,
                    groups=None) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    mask = np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    out = {
        "auc":   float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "f1":    float(f1_score(y_true, y_score >= threshold, zero_division=0)),
        "n":     int(len(y_true)),
        "pos_rate": float(y_true.mean()),
    }
    if groups is not None:
        out["auc_strat"], out["n_strata"] = stratified_auc(y_true, y_score, np.asarray(groups)[mask])
    return out


def print_metrics(name: str, metrics: dict) -> None:
    strat = (f" AUC_strat={metrics['auc_strat']:.4f} ({metrics['n_strata']} strata)"
             if "auc_strat" in metrics else "")
    print(f"[{name}] AUC={metrics['auc']:.4f}{strat} AUPRC={metrics['auprc']:.4f} "
          f"F1={metrics['f1']:.4f} n={metrics['n']:,} pos={metrics['pos_rate']:.1%}")


def save_metrics(path, results: dict) -> None:
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
