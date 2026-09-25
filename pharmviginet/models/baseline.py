#!/usr/bin/env python3
"""
Baseline signal detector — evaluates classical ROR and PRR against labels.

Usage:
    python -m pharmviginet.models.baseline          # val + test
    python -m pharmviginet.models.baseline --split val
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd

from pharmviginet.config import LOGS, VAL_PARQUET, TEST_PARQUET
from pharmviginet.utils.metrics import compute_metrics, print_metrics, save_metrics

COLS = ["label", "ror", "ror_lower_ci", "n_reports", "drugname", "pt", "ror_train"]


def load_split(path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=COLS)
    df = df.dropna(subset=["ror", "label"])
    df["ror"] = pd.to_numeric(df["ror"], errors="coerce")
    df["label"] = df["label"].astype(int)
    return df


def compute_prr(df: pd.DataFrame) -> pd.Series:
    """PRR = (a/n_drug) / (c/N) per (drugname, pt) pair."""
    N = df["drugname"].nunique()
    drug_counts = df.groupby("drugname")["label"].count().rename("n_drug")
    pair_counts = df.groupby(["drugname", "pt"])["label"].count().rename("a")
    reac_counts = df.groupby("pt")["label"].count().rename("n_reac")

    prr_df = (pair_counts
              .reset_index()
              .merge(drug_counts.reset_index(), on="drugname")
              .merge(reac_counts.reset_index(), on="pt"))
    prr_df["prr"] = (prr_df["a"] / prr_df["n_drug"]) / (prr_df["n_reac"] / N + 1e-9)

    return df.merge(prr_df[["drugname", "pt", "prr"]], on=["drugname", "pt"], how="left")["prr"]


def evaluate(split_name: str, path) -> dict:
    print(f"Loading {split_name} …")
    df = load_split(path)
    print(f"  {len(df):,} rows, {df['label'].mean():.1%} positive")

    y_true = df["label"].values

    # ROR_all (cheating baseline — label derived from this)
    ror_score = np.log1p(df["ror"].clip(lower=0).values)
    ror_metrics = compute_metrics(y_true, ror_score)
    print_metrics(f"{split_name}/ROR_all", ror_metrics)

    # ROR_train (fair baseline — computed on train set only, NaN for new pairs)
    ror_train_raw = pd.to_numeric(df["ror_train"], errors="coerce")
    ror_train_score = np.log1p(ror_train_raw.clip(lower=0).fillna(0).values)
    ror_train_metrics = compute_metrics(y_true, ror_train_score)
    n_new = ror_train_raw.isna().sum()
    print_metrics(f"{split_name}/ROR_train", ror_train_metrics)
    print(f"  ({n_new:,} new pairs scored 0 — model advantage zone)")

    # PRR score
    prr_score = np.log1p(compute_prr(df).clip(lower=0).fillna(0).values)
    prr_metrics = compute_metrics(y_true, prr_score)
    print_metrics(f"{split_name}/PRR", prr_metrics)

    return {"ror_all": ror_metrics, "ror_train": ror_train_metrics, "prr": prr_metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    args = parser.parse_args()

    results = {}
    if args.split in ("val", "both"):
        results["val"] = evaluate("val", VAL_PARQUET)
    if args.split in ("test", "both"):
        results["test"] = evaluate("test", TEST_PARQUET)

    out = LOGS / "baseline_results.json"
    save_metrics(out, results)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
