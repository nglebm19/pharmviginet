#!/usr/bin/env python3
"""
Baseline signal detector — evaluates classical ROR and PRR against labels.

Usage:
    python -m pharmviginet.models.baseline          # val + test
    python -m pharmviginet.models.baseline --split val
    python -m pharmviginet.models.baseline --labels sider   # independent SIDER labels
    python -m pharmviginet.models.baseline --level row      # old per-report scoring
    python -m pharmviginet.models.baseline --task a         # Task A: ingredient-disjoint folds

--task timesplit (default) is the legacy time-split reference. It is not Task A:
ROR_all and within-split PRR use eval-period data (FB-D11). ROR_train is computed
here from train-split counts; the split files carry no ror_train column (FB-D13).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import pyarrow.parquet as pq

from pharmviginet.config import (
    FEATURE_CUTOFF_YEAR, LABELS_SIDER, LOGS, TASK_A_MIN_REPORTS, TASK_A_N_FOLDS, TASK_A_PAIRS,
    TASK_A_RESULTS, TASK_A_SEED, TEST_PARQUET, TRAIN_PARQUET, VAL_PARQUET,
)
from pharmviginet.models import drug_control
from pharmviginet.models.disproportionality import (
    bcpnn, compute_prr, ebgm, fit_mgps_prior, pair_counts, prr, ror,
)
from pharmviginet.utils.metrics import compute_metrics, fold_metrics, print_metrics, save_metrics

COLS = ["label", "ror", "ror_lower_ci", "n_reports", "drugname", "pt", "primaryid"]
PAIR = ["drugname", "pt"]


def train_counts() -> tuple[pd.DataFrame, tuple]:
    """Train-period pair counts + fitted MGPS prior (fair: no future data)."""
    print("Counting train-period reports …")
    cols = ["primaryid", "drugname", "pt"]
    df = pq.read_table(TRAIN_PARQUET, columns=cols, read_dictionary=cols).to_pandas()
    c = pair_counts(df)
    del df
    c["drugname"] = c["drugname"].astype(str)
    c["pt"] = c["pt"].astype(str)
    c["prr_train"] = prr(c)
    c["ror_train"] = ror(c)
    c["ic"], c["ic025"] = bcpnn(c["a"], c["E"])
    prior = fit_mgps_prior(c["a"].values, c["E"].values)
    print(f"  {len(c):,} train pairs; MGPS prior (a1,b1,a2,b2,p) = "
          + ", ".join(f"{v:.4g}" for v in prior))
    return c[PAIR + ["a", "E", "prr_train", "ror_train", "ic", "ic025"]], prior


def add_train_scores(df: pd.DataFrame, counts: pd.DataFrame, prior: tuple) -> pd.DataFrame:
    """Join train-period scores; pairs unseen in train get neutral scores."""
    df = df.merge(counts, on=PAIR, how="left")
    seen = df["a"].notna()
    df["ebgm"], df["eb05"] = 1.0, 1.0
    if seen.any():
        g, g05 = ebgm(df.loc[seen, "a"].values, df.loc[seen, "E"].values, prior)
        df.loc[seen, "ebgm"], df.loc[seen, "eb05"] = g, g05
    return df.fillna({"prr_train": 1.0, "ic": 0.0, "ic025": 0.0})


def load_split(path, labels: str = "ror", level: str = "pair") -> pd.DataFrame:
    df = pd.read_parquet(path, columns=COLS)
    df = df.merge(compute_prr(df), on=PAIR, how="left")
    if labels == "sider":
        sider = pd.read_parquet(LABELS_SIDER, columns=PAIR + ["label"])
        df = df.drop(columns="label").merge(sider, on=PAIR, how="inner")
    df = df.dropna(subset=["ror", "label"])
    if level == "pair":
        # ror / ror_train / prr / label are constant within a pair
        df = df.drop_duplicates(subset=PAIR)
    df["ror"] = pd.to_numeric(df["ror"], errors="coerce")
    df["label"] = df["label"].astype(int)
    return df


def evaluate(split_name: str, path, labels: str = "ror", level: str = "pair",
             train: tuple | None = None) -> dict:
    print(f"Loading {split_name} ({labels} labels, {level} level) …")
    df = load_split(path, labels, level)
    if train is not None:
        df = add_train_scores(df, *train)
    print(f"  {len(df):,} {level}s, {df['label'].mean():.1%} positive")

    y_true = df["label"].values
    groups = df["pt"].values

    # ROR_all — circular under ror labels (label derived from it)
    ror_score = np.log1p(df["ror"].clip(lower=0).values)
    ror_metrics = compute_metrics(y_true, ror_score, groups=groups)
    print_metrics(f"{split_name}/ROR_all", ror_metrics)

    # ROR_train (fair baseline — computed on train set only, NaN for new pairs)
    ror_train_raw = pd.to_numeric(df["ror_train"], errors="coerce")
    ror_train_score = np.log1p(ror_train_raw.clip(lower=0).fillna(0).values)
    ror_train_metrics = compute_metrics(y_true, ror_train_score, groups=groups)
    n_new = ror_train_raw.isna().sum()
    print_metrics(f"{split_name}/ROR_train", ror_train_metrics)
    print(f"  ({n_new:,} new pairs scored 0 — model advantage zone)")

    # PRR (computed within this split)
    prr_score = np.log1p(df["prr"].clip(lower=0).fillna(0).values)
    prr_metrics = compute_metrics(y_true, prr_score, groups=groups)
    print_metrics(f"{split_name}/PRR", prr_metrics)

    results = {"ror_all": ror_metrics, "ror_train": ror_train_metrics, "prr": prr_metrics}
    if train is not None:
        # train-period methods; log-scale where the statistic is a ratio
        for name, score in [("prr_train", np.log(df["prr_train"].clip(lower=1e-9))),
                            ("ic", df["ic"]), ("ic025", df["ic025"]),
                            ("ebgm", np.log(df["ebgm"])), ("eb05", np.log(df["eb05"]))]:
            m = compute_metrics(y_true, score.values, threshold=0.0, groups=groups)
            print_metrics(f"{split_name}/{name.upper()}", m)
            results[name] = m
    return results


def evaluate_task_a() -> dict:
    """Classical scores + drug-level control on ingredient-disjoint folds (built by data.task_a)."""
    df = pd.read_parquet(TASK_A_PAIRS)
    print(f"Task A: {len(df):,} pairs, {df['label'].mean():.1%} positive, {df['fold'].nunique()} folds")
    # log-scale where the statistic is a ratio, as in the time-split evaluation
    df["s_ror"] = np.log(df["ror"])
    df["s_prr"] = np.log(df["prr"])
    df["s_ic"], df["s_ic025"] = df["ic"], df["ic025"]
    df["s_ebgm"], df["s_eb05"] = np.log(df["ebgm"]), np.log(df["eb05"])
    print("Fitting drug-level control (out-of-fold) …")
    df["s_drug_control"] = drug_control.oof_scores(df)
    cols = ["s_ror", "s_prr", "s_ic", "s_ic025", "s_ebgm", "s_eb05", "s_drug_control"]
    res = fold_metrics(df, cols, ref="s_ic")

    # rename s_x → x in the output
    strip = lambda d: {k[2:]: v for k, v in d.items()}
    res = {"ref": "ic", "folds": {k: strip(v) for k, v in res["folds"].items()},
           "summary": strip(res["summary"])}
    prior = json.loads(TASK_A_PAIRS.with_suffix(".prior.json").read_text())
    res["config"] = {
        "feature_cutoff_year": FEATURE_CUTOFF_YEAR, "min_reports": TASK_A_MIN_REPORTS,
        "n_folds": TASK_A_N_FOLDS, "seed": TASK_A_SEED, "n_pairs": int(len(df)),
        "pos_rate": float(df["label"].mean()), "mgps_prior": prior,
    }
    print(f"\n{'method':14s} {'AUC_strat mean ± sd':>22s} {'diff vs IC mean ± sd':>24s}")
    for m, v in sorted(res["summary"].items(), key=lambda kv: -kv[1]["auc_strat_mean"]):
        print(f"{m:14s} {v['auc_strat_mean']:>14.4f} ± {v['auc_strat_sd']:.4f} "
              f"{v['diff_vs_ref_mean']:>+15.4f} ± {v['diff_vs_ref_sd']:.4f}")
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["timesplit", "a"], default="timesplit",
                        help="timesplit = legacy reference; a = Task A ingredient-disjoint folds")
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    parser.add_argument("--labels", choices=["ror", "sider"], default="ror")
    parser.add_argument("--level", choices=["pair", "row"], default="pair",
                        help="score each unique (drug, pt) pair once, or every report row")
    args = parser.parse_args()

    if args.task == "a":
        save_metrics(TASK_A_RESULTS, evaluate_task_a())
        print(f"\nSaved → {TASK_A_RESULTS}")
        return

    train = train_counts()
    results = {}
    if args.split in ("val", "both"):
        results["val"] = evaluate("val", VAL_PARQUET, args.labels, args.level, train)
    if args.split in ("test", "both"):
        results["test"] = evaluate("test", TEST_PARQUET, args.labels, args.level, train)

    suffix = ("" if args.labels == "ror" else f"_{args.labels}") + ("" if args.level == "pair" else "_row")
    out = LOGS / f"baseline_results{suffix}.json"
    save_metrics(out, results)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
