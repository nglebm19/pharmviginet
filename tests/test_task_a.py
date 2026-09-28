import numpy as np
import pandas as pd
import pytest

from pharmviginet.data import task_a
from pharmviginet.models import drug_control
from pharmviginet.utils.metrics import fold_metrics


def _reports():
    # D+E in reports 1-3 (2021-2022) and report 4 (2023); X+E in report 5; X+O in 6-8
    rows = [(1, "D", "E", 2021), (2, "D", "E", 2022), (3, "D", "E", 2022), (4, "D", "E", 2023),
            (5, "X", "E", 2020), (6, "X", "O", 2020), (7, "X", "O", 2020), (8, "X", "O", 2023)]
    return pd.DataFrame(rows, columns=["primaryid", "drugname", "pt", "year"])


def test_counts_ignore_reports_after_cutoff():
    c, drugs = task_a._counts_from_df(_reports(), cutoff=2022)
    row = c.set_index(["drugname", "pt"]).loc[("D", "E")]
    assert (row["a"], row["n_drug"], row["n_reac"], row["N"]) == (3, 3, 4, 6)
    d = drugs.set_index("drugname")
    assert d.loc["X", "n_pt"] == 2 and d.loc["D", "first_year"] == 2021


def test_eligible_pairs_uses_cutoff_counts_only():
    c, _ = task_a._counts_from_df(_reports(), cutoff=2022)
    labels = pd.DataFrame({"drugname": ["D", "X"], "pt": ["E", "O"], "ingredients": ["d", "x"],
                           "label": [1, 0], "n_reports": [99, 99]})
    out = task_a.eligible_pairs(c, labels, min_reports=3)
    # X+O has 2 reports up to 2022 (3 over all years) → not eligible
    assert list(zip(out["drugname"], out["pt"])) == [("D", "E")]
    assert "n_reports" not in out.columns


def _pairs():
    ings = ["a"] * 30 + ["b"] * 20 + ["c"] * 20 + ["d"] * 10 + ["a|b"] * 5 + ["a|a2"] * 3 + ["a2"] * 12
    return pd.DataFrame({"drugname": [f"N{i % 7}_{s}" for i, s in enumerate(ings)],
                         "pt": [f"P{i % 4}" for i in range(len(ings))],
                         "ingredients": ings, "label": [i % 3 == 0 for i in range(len(ings))]})


def test_folds_are_ingredient_disjoint_and_deterministic():
    p = _pairs()
    f1 = task_a.assign_folds(p["ingredients"], n_folds=3, seed=1)
    f2 = task_a.assign_folds(p["ingredients"], n_folds=3, seed=1)
    pd.testing.assert_frame_equal(f1, f2)
    assert f1["ingredient"].is_unique
    out, dropped = task_a.pair_folds(p, f1)
    fmap = dict(zip(f1["ingredient"], f1["fold"]))
    for ings, k in zip(out["ingredients"], out["fold"]):
        assert all(fmap[x] == k for x in ings.split("|"))
    # every ingredient set lands in exactly one fold (synonym drugnames share it)
    assert out.groupby("ingredients")["fold"].nunique().max() == 1
    spanning = p["ingredients"].map(lambda s: len({fmap[x] for x in s.split("|")}) > 1).sum()
    assert dropped == spanning and len(out) == len(p) - dropped


def test_fold_quality_flags_imbalance():
    p = pd.DataFrame({"fold": [0] * 100 + [1] * 10, "label": [0, 1] * 55,
                      "ingredients": ["a"] * 100 + ["b"] * 10, "pt": ["P"] * 110})
    q, bad = task_a.fold_quality(p)
    assert q.loc[0, "pairs"] == 100 and q.loc[1, "strata"] == 0
    assert any("ratio" in b for b in bad) and any("strata" in b for b in bad)


def test_fold_metrics_mean_sd_and_paired_diff():
    rows = []
    for k, flip in [(0, False), (1, True)]:
        y = [0] * 10 + [1] * 10
        good = list(range(20))
        rows += [{"fold": k, "pt": "P", "label": yi, "ref": (-s if flip else s), "m": s}
                 for yi, s in zip(y, good)]
    res = fold_metrics(pd.DataFrame(rows), ["ref", "m"], ref="ref", min_n=20)
    assert res["folds"][0]["ref"]["auc_strat"] == 1.0 and res["folds"][1]["ref"]["auc_strat"] == 0.0
    assert res["summary"]["ref"]["auc_strat_mean"] == pytest.approx(0.5)
    assert res["summary"]["ref"]["auc_strat_sd"] == pytest.approx(np.std([1.0, 0.0], ddof=1))
    assert res["summary"]["m"]["diff_vs_ref_mean"] == pytest.approx(0.5)
    assert all(res["folds"][k][c]["n"] == 20 for k in (0, 1) for c in ("ref", "m"))


def test_fold_metrics_rejects_non_finite():
    df = pd.DataFrame({"fold": [0] * 4, "pt": ["P"] * 4, "label": [0, 1, 0, 1],
                       "ic": [0.1, np.nan, 0.3, 0.4]})
    with pytest.raises(ValueError):
        fold_metrics(df, ["ic"])


def test_drug_control_is_out_of_fold():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({f: rng.normal(size=n) for f in drug_control.FEATURES})
    df["fold"] = np.arange(n) % 3
    df["label"] = (df["log_n_drug"] > 0).astype(int)
    before = drug_control.oof_scores(df)
    flipped = df.copy()
    flipped.loc[flipped["fold"] == 0, "label"] ^= 1
    after = drug_control.oof_scores(flipped)
    m = df["fold"].to_numpy() == 0
    np.testing.assert_allclose(before[m], after[m])
    assert not (set(drug_control.FEATURES) & {"pt", "a", "E", "ic", "ror", "prr", "ebgm"})
