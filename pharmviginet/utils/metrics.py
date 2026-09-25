import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    mask = np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    return {
        "auc":   float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "f1":    float(f1_score(y_true, y_score >= threshold, zero_division=0)),
        "n":     int(len(y_true)),
        "pos_rate": float(y_true.mean()),
    }


def print_metrics(name: str, metrics: dict) -> None:
    print(f"[{name}] AUC={metrics['auc']:.4f} AUPRC={metrics['auprc']:.4f} "
          f"F1={metrics['f1']:.4f} n={metrics['n']:,} pos={metrics['pos_rate']:.1%}")


def save_metrics(path, results: dict) -> None:
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
