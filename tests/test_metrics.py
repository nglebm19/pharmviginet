import numpy as np
import pandas as pd
import pytest

from pharmviginet.models import baseline
from pharmviginet.utils.metrics import compute_metrics, stratified_auc


def test_stratified_auc_weighted_mean_and_skips():
    # group A: perfect ranking (AUC 1.0), 20 rows
    ya = [0] * 10 + [1] * 10
    sa = list(range(20))
    # group B: inverted ranking (AUC 0.0), 30 rows
    yb = [1] * 15 + [0] * 15
    sb = list(range(30))
    # group C: single class → skipped; group D: too small → skipped
    yc, sc = [1] * 25, list(range(25))
    yd, sd = [0, 1], [0, 1]
    y = ya + yb + yc + yd
    s = sa + sb + sc + sd
    g = ["A"] * 20 + ["B"] * 30 + ["C"] * 25 + ["D"] * 2
    auc, n = stratified_auc(y, s, g, min_n=20)
    assert n == 2
    assert auc == pytest.approx((1.0 * 20 + 0.0 * 30) / 50)


def test_compute_metrics_groups_optional():
    y = np.array([0, 1] * 20)
    s = np.arange(40, dtype=float)
    assert "auc_strat" not in compute_metrics(y, s)
    m = compute_metrics(y, s, groups=np.array(["x"] * 40))
    assert m["n_strata"] == 1


def test_compute_prr():
    # N = 10 reports. Drug D in reports 1-4; event E in reports 1,2 (with D) and 5 (without D).
    rows = [(i, "D", "E") for i in (1, 2)] + [(i, "D", "OTHER") for i in (3, 4)]
    rows += [(5, "X", "E")] + [(i, "X", "OTHER") for i in range(6, 11)]
    rows += [(1, "D", "E")]  # duplicate row must not double count
    df = pd.DataFrame(rows, columns=["primaryid", "drugname", "pt"])
    prr = baseline.compute_prr(df).set_index(["drugname", "pt"])["prr"]
    # a=2, n_drug=4, n_reac=3, N=10 → (2/4) / ((3-2)/(10-4)) = 3.0
    assert prr[("D", "E")] == pytest.approx(3.0)


def test_load_split_pair_level_dedups(tmp_path):
    df = pd.DataFrame({
        "label": [1, 1, 0], "ror": [2.0, 2.0, 0.5], "ror_lower_ci": [1.1, 1.1, 0.2],
        "n_reports": [2, 2, 1], "drugname": ["D", "D", "D"], "pt": ["E", "E", "F"],
        "ror_train": [2.0, 2.0, np.nan], "primaryid": ["1", "2", "3"],
    })
    path = tmp_path / "split.parquet"
    df.to_parquet(path)
    assert len(baseline.load_split(path, level="pair")) == 2
    assert len(baseline.load_split(path, level="row")) == 3
