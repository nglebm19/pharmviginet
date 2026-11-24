"""Re-score Task A baselines on Method F folds, read-only on production artifacts.

Run from the repository root (needs local data; ~8 min, ~3 GB peak):
    python -m experiments.graph_folds.evaluate --output experiments/graph_folds/results.json

Steps
1. Rebuild Task A eligible pairs exactly as `task_a.main` does (counts up to 2022,
   >= 3 reports, SIDER labels) and score them with the *saved* MGPS prior, so pairs
   shared with production get identical scores (asserted).
2. Replay the production evaluation on `task_a_pairs.parquet`; must equal
   `data/logs/task_a_results.json` (asserted) — this validates the wrapper below.
3. Score scenarios: production pairs with F folds; all eligible pairs with F folds
   (subsets: single-ingredient, combination, recovered); seeded F variants.

Writes only the --output JSON: aggregate metrics, counts and fold-quality
summaries. No pair rows, no labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.graph_folds.components import component_folds, graph_summary
from pharmviginet import config as c
from pharmviginet.data import task_a
from pharmviginet.models import drug_control
from pharmviginet.utils.metrics import fold_metrics

METHODS = ["ror", "prr", "ic", "ic025", "ebgm", "eb05", "drug_control"]
SCORES = ["ror", "prr", "ic", "ic025", "ebgm", "eb05"]


def add_method_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Same transforms and control as `baseline.evaluate_task_a`."""
    df = df.copy()
    df["s_ror"], df["s_prr"] = np.log(df["ror"]), np.log(df["prr"])
    df["s_ic"], df["s_ic025"] = df["ic"], df["ic025"]
    df["s_ebgm"], df["s_eb05"] = np.log(df["ebgm"]), np.log(df["eb05"])
    df["s_drug_control"] = drug_control.oof_scores(df)
    return df


def summarize(df: pd.DataFrame) -> dict:
    res = fold_metrics(df, [f"s_{m}" for m in METHODS], ref="s_ic")
    strip = lambda d: {k[2:]: v for k, v in d.items()}
    return {"ref": "ic", "folds": {k: strip(v) for k, v in res["folds"].items()},
            "summary": strip(res["summary"])}


def leakage_checks(df: pd.DataFrame, folds: pd.DataFrame) -> dict:
    fmap = dict(zip(folds["ingredient"], folds["fold"]))
    spans = df["ingredients"].map(lambda s: len({fmap[i] for i in s.split("|")}))
    ing = df.assign(i=df["ingredients"].str.split("|")).explode("i")
    out = {
        "ingredients_in_multiple_folds": int((ing.groupby("i")["fold"].nunique() > 1).sum()),
        "drugnames_in_multiple_folds": int((df.groupby("drugname")["fold"].nunique() > 1).sum()),
        "pairs_spanning_folds": int((spans > 1).sum()),
        "pair_fold_mismatch": int((df["ingredients"].map(lambda s: fmap[s.split("|")[0]]) != df["fold"]).sum()),
    }
    assert not any(out.values()), out
    return out


def quality(df: pd.DataFrame) -> dict:
    q, bad = task_a.fold_quality(df)  # labels used only to validate, never to assign
    return {"ok": not bad, "breaches": bad,
            "pairs_ratio": float(q["pairs"].max() / q["pairs"].min()),
            "pairs_min": int(q["pairs"].min()), "pairs_max": int(q["pairs"].max()),
            "pos_rate_min": float(q["pos_rate"].min()), "pos_rate_max": float(q["pos_rate"].max()),
            "strata_min": int(q["strata"].min()), "top1_share_max": float(q["top1_share"].max()),
            "ingredients_per_fold": q["ingredients"].astype(int).tolist()}


def scenario(df: pd.DataFrame, folds: pd.DataFrame, subsets: dict | None = None) -> dict:
    df, dropped = task_a.pair_folds(df.drop(columns="fold", errors="ignore"), folds)
    assert dropped == 0
    out = {"n_pairs": len(df), "pos_rate": float(df["label"].mean()),
           "leakage": leakage_checks(df, folds), "quality": quality(df)}
    scored = add_method_scores(df)
    out["all"] = summarize(scored)
    for name, mask in (subsets or {}).items():
        sub = scored[mask(scored)]
        out[name] = {"n_pairs": len(sub), "pos_rate": float(sub["label"].mean()), **summarize(sub)}
    return out


def main(output: Path, seeds: int) -> None:
    saved = json.loads(c.TASK_A_RESULTS.read_text())
    prior_d = json.loads(c.TASK_A_PAIRS.with_suffix(".prior.json").read_text())
    prior = tuple(prior_d[k] for k in ["a1", "b1", "a2", "b2", "p"])
    prod = pd.read_parquet(c.TASK_A_PAIRS)
    prod_folds = pd.read_parquet(c.TASK_A_FOLDS)

    print("1. eligible pairs from master (<= 2022) + saved prior", flush=True)
    counts, drugs = task_a.feature_counts()
    labels = pd.read_parquet(c.LABELS_SIDER, columns=task_a.LABEL_COLS)
    eligible = task_a.eligible_pairs(counts, labels)
    del counts
    eligible = task_a.add_scores(eligible, prior).merge(drugs, on="drugname", how="left")
    eligible["n_ingredients"] = eligible["ingredients"].str.count(r"\|") + 1
    keep = task_a.PAIR + ["ingredients", "label", "a", "n_drug", "n_reac", "N", "E"] + SCORES + task_a.DRUG_FEATURES
    eligible = eligible[keep]
    # Shared pairs must match production. EBGM/EB05 differ by ~1e-8 relative because the
    # saved prior JSON is rounded (same tolerance as scripts/audit_results --counts); PRR's
    # fill value is the max finite PRR over the scored set and can differ.
    shared = prod.merge(eligible, on=task_a.PAIR, suffixes=("", "_e"))
    assert len(shared) == len(prod)
    for col in ["label", "a", "E", "ror", "ic", "ic025", "ebgm", "eb05"] + task_a.DRUG_FEATURES:
        np.testing.assert_allclose(shared[col].astype(float), shared[f"{col}_e"].astype(float),
                                   rtol=1e-7, atol=1e-9, err_msg=col)
    prr_diff = int((~np.isclose(shared["prr"], shared["prr_e"])).sum())
    # Then use production's stored scores for shared pairs, so scenarios on production
    # pairs differ from production only in the folds.
    eligible = eligible.set_index(task_a.PAIR)
    eligible.update(prod.set_index(task_a.PAIR)[SCORES])
    eligible = eligible.reset_index()

    print("2. replay production evaluation", flush=True)
    prod_scored = add_method_scores(prod)
    replay = summarize(prod_scored)
    for m in METHODS:
        for k in ["auc_strat_mean", "auc_strat_sd", "diff_vs_ref_mean"]:
            np.testing.assert_allclose(replay["summary"][m][k], saved["summary"][m][k], rtol=1e-9)

    print("3. Method F scenarios", flush=True)
    f_folds = component_folds(eligible["ingredients"])
    is_prod = eligible.set_index(task_a.PAIR).index.isin(prod.set_index(task_a.PAIR).index)
    combo = lambda d: d["n_ingredients"] > 1
    prod_keys = set(zip(prod["drugname"], prod["pt"]))
    subsets = {"single_ingredient": lambda d: d["n_ingredients"] == 1,
               "combination": combo,
               "recovered": lambda d: ~pd.Series(list(zip(d["drugname"], d["pt"])), index=d.index).isin(prod_keys)}
    prod_subsets = {"single_ingredient": lambda d: d["n_ingredients"] == 1, "combination": combo}

    results = {
        "graph": graph_summary(eligible["ingredients"]),
        "eligible_pairs": len(eligible), "production_pairs": len(prod),
        "prr_fill_differences_on_shared_pairs": prr_diff,
        "A_production_greedy": {"n_pairs": len(prod), "quality": quality(prod),
                                "leakage": leakage_checks(prod, prod_folds),
                                "all": replay, **{k: {"n_pairs": int(m(prod_scored).sum()), **summarize(
                                    prod_scored[m(prod_scored)])} for k, m in prod_subsets.items()}},
        "F_production_pairs": scenario(eligible[is_prod], component_folds(eligible.loc[is_prod, "ingredients"]),
                                       prod_subsets),
        "F_all_pairs": scenario(eligible, f_folds, subsets),
        "F_seeded_all_pairs": {},
    }
    for seed in range(1, seeds + 1):
        print(f"   seeded F variant {seed}", flush=True)
        r = scenario(eligible, component_folds(eligible["ingredients"], seed=seed))
        results["F_seeded_all_pairs"][seed] = {"quality": r["quality"], "summary": r["all"]["summary"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, default=float) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    main(a.output, a.seeds)
