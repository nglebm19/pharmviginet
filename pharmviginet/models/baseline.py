#!/usr/bin/env python3
"""
Baseline signal detector — evaluates classical ROR and PRR against labels.

Usage:
    python -m pharmviginet.models.baseline          # val + test
    python -m pharmviginet.models.baseline --split val
    python -m pharmviginet.models.baseline --labels sider   # independent SIDER labels
    python -m pharmviginet.models.baseline --level row      # old per-report scoring
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from pharmviginet.config import LABELS_SIDER, LOGS, VAL_PARQUET, TEST_PARQUET
from pharmviginet.utils.metrics import compute_metrics, print_metrics, save_metrics

COLS = ["label", "ror", "ror_lower_ci", "n_reports", "drugname", "pt", "ror_train", "primaryid"]
PAIR = ["drugname", "pt"]


def compute_prr(df: pd.DataFrame) -> pd.DataFrame:
    """
    PRR per (drugname, pt), counting distinct reports (primaryid):
        PRR = (a / n_drug) / ((n_reac - a) / (N - n_drug))
    Pass all rows of the split — before label filtering — so counts are complete.
    """
    rep = df[["primaryid", "drugname", "pt"]].drop_duplicates()
    N = rep["primaryid"].nunique()
    a = rep.groupby(PAIR)["primaryid"].nunique().rename("a").reset_index()
    n_drug = rep.groupby("drugname")["primaryid"].nunique().rename("n_drug").reset_index()
    n_reac = rep.groupby("pt")["primaryid"].nunique().rename("n_reac").reset_index()
    p = a.merge(n_drug, on="drugname").merge(n_reac, on="pt")
    denom = (p["n_reac"] - p["a"]) / (N - p["n_drug"])
    p["prr"] = (p["a"] / p["n_drug"]) / denom.replace(0, np.nan)
    return p[PAIR + ["prr"]]


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


def evaluate(split_name: str, path, labels: str = "ror", level: str = "pair") -> dict:
    print(f"Loading {split_name} ({labels} labels, {level} level) …")
    df = load_split(path, labels, level)
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

    return {"ror_all": ror_metrics, "ror_train": ror_train_metrics, "prr": prr_metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    parser.add_argument("--labels", choices=["ror", "sider"], default="ror")
    parser.add_argument("--level", choices=["pair", "row"], default="pair",
                        help="score each unique (drug, pt) pair once, or every report row")
    args = parser.parse_args()

    results = {}
    if args.split in ("val", "both"):
        results["val"] = evaluate("val", VAL_PARQUET, args.labels, args.level)
    if args.split in ("test", "both"):
        results["test"] = evaluate("test", TEST_PARQUET, args.labels, args.level)

    suffix = ("" if args.labels == "ror" else f"_{args.labels}") + ("" if args.level == "pair" else "_row")
    out = LOGS / f"baseline_results{suffix}.json"
    save_metrics(out, results)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
