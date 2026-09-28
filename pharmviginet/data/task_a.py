#!/usr/bin/env python3
"""
Task A dataset — SIDER labels, features from reports up to FEATURE_CUTOFF_YEAR,
ingredient-disjoint folds.

Each RxNorm ingredient is assigned to one fold. For fold k, the ingredients in k
are label-held-out ingredients: their FAERS features (<= cutoff) are available,
but their SIDER labels are not used for training. The same PT can and does
appear in every fold.

Eligibility: >= TASK_A_MIN_REPORTS distinct reports up to the cutoff. Only
drugname, pt, ingredients and label are taken from labels_sider; its all-years
n_reports column is not used. (labels_sider keeps pairs with >= 3 reports over
all years, a superset of pairs with >= 3 reports up to the cutoff.)

Usage:
    python -m pharmviginet.data.task_a
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pharmviginet.config import (
    FEATURE_CUTOFF_YEAR, LABELS_SIDER, MASTER_PARQUET, TASK_A_FOLDS, TASK_A_MIN_REPORTS,
    TASK_A_N_FOLDS, TASK_A_PAIRS, TASK_A_SEED,
)
from pharmviginet.models.disproportionality import (
    bcpnn, ebgm, fit_mgps_prior, pair_counts, prr, ror,
)

PAIR = ["drugname", "pt"]
LABEL_COLS = PAIR + ["ingredients", "label"]
DRUG_FEATURES = ["log_n_drug", "n_pt", "first_year", "n_ingredients"]

# fold-quality limits; any breach stops the pipeline
MAX_PAIR_RATIO = 1.5      # largest / smallest fold, in pairs
MAX_POS_GAP = 0.05        # |fold positive rate - overall|
MIN_STRATA = 500          # usable PT strata per fold
MAX_TOP1_SHARE = 0.10     # share of a fold's pairs held by its largest ingredient
MIN_STRATUM_N = 20        # same rule as stratified_auc


def _counts_from_df(df: pd.DataFrame, cutoff: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """[primaryid, drugname, pt, year] rows → (pair counts, drug-level features), <= cutoff only."""
    df = df[df["year"] <= cutoff]
    c = pair_counts(df)
    c["drugname"] = c["drugname"].astype(str)
    c["pt"] = c["pt"].astype(str)
    drugs = pd.DataFrame({
        "first_year": df.groupby("drugname", observed=True)["year"].min(),
    })
    drugs.index = drugs.index.astype(str)
    drugs["n_pt"] = c.groupby("drugname").size()
    drugs["log_n_drug"] = np.log1p(c.groupby("drugname")["n_drug"].first())
    return c, drugs.rename_axis("drugname").reset_index()


def feature_counts(cutoff: int = FEATURE_CUTOFF_YEAR) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["primaryid", "drugname", "pt", "year"]
    df = pq.read_table(MASTER_PARQUET, columns=cols, filters=[("year", "<=", cutoff)],
                       read_dictionary=["primaryid", "drugname", "pt"]).to_pandas()
    return _counts_from_df(df, cutoff)


def eligible_pairs(counts: pd.DataFrame, labels: pd.DataFrame,
                   min_reports: int = TASK_A_MIN_REPORTS) -> pd.DataFrame:
    """Pairs with >= min_reports reports up to the cutoff that carry a SIDER label."""
    c = counts[counts["a"] >= min_reports]
    return c.merge(labels[LABEL_COLS], on=PAIR, how="inner")


def assign_folds(ingredients: pd.Series, n_folds: int = TASK_A_N_FOLDS,
                 seed: int = TASK_A_SEED) -> pd.DataFrame:
    """
    ingredients: '|'-joined ingredient set, one entry per pair.
    Shuffle unique ingredients, then give each to the fold with the fewest pairs so far
    (pair weight = number of pairs containing the ingredient). → [ingredient, fold]
    """
    weight = ingredients.str.split("|").explode().value_counts().sort_index()
    order = np.random.default_rng(seed).permutation(len(weight))
    load = np.zeros(n_folds)
    rows = []
    for i in order:
        k = int(np.argmin(load))
        load[k] += weight.iloc[i]
        rows.append((weight.index[i], k))
    return pd.DataFrame(rows, columns=["ingredient", "fold"]).sort_values("ingredient", ignore_index=True)


def pair_folds(pairs: pd.DataFrame, folds: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Give each pair its ingredients' fold; drop pairs whose ingredients span folds."""
    fmap = dict(zip(folds["ingredient"], folds["fold"]))
    sets = pairs["ingredients"].map(lambda s: {fmap[x] for x in s.split("|")})
    one = sets.map(len).eq(1)
    out = pairs[one].copy()
    out["fold"] = sets[one].map(lambda s: next(iter(s))).astype(int)
    return out, int((~one).sum())


def fold_quality(pairs: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Per-fold report and a list of limit breaches (empty = OK)."""
    overall = pairs["label"].mean()
    rows = []
    for k, d in pairs.groupby("fold"):
        per_ing = d["ingredients"].str.split("|").explode().value_counts()
        g = d.groupby("pt")["label"].agg(["size", "nunique"])
        rows.append({
            "fold": k, "pairs": len(d), "pos_rate": d["label"].mean(),
            "ingredients": len(per_ing),
            "strata": int(((g["size"] >= MIN_STRATUM_N) & (g["nunique"] == 2)).sum()),
            "top1_ingredient": per_ing.index[0], "top1_share": per_ing.iloc[0] / len(d),
            "top10_share": per_ing.head(10).sum() / len(d),
        })
    q = pd.DataFrame(rows).set_index("fold")
    bad = []
    if q["pairs"].max() / q["pairs"].min() > MAX_PAIR_RATIO:
        bad.append(f"pair count ratio {q['pairs'].max() / q['pairs'].min():.2f} > {MAX_PAIR_RATIO}")
    for k, r in q.iterrows():
        if abs(r["pos_rate"] - overall) > MAX_POS_GAP:
            bad.append(f"fold {k}: positive rate {r['pos_rate']:.1%} vs overall {overall:.1%}")
        if r["strata"] < MIN_STRATA:
            bad.append(f"fold {k}: {r['strata']} usable strata < {MIN_STRATA}")
        if r["top1_share"] > MAX_TOP1_SHARE:
            bad.append(f"fold {k}: {r['top1_ingredient']} holds {r['top1_share']:.1%} of pairs")
    return q, bad


def add_scores(pairs: pd.DataFrame, prior: tuple) -> pd.DataFrame:
    p = pairs.copy()
    p["ror"] = ror(p)
    p["prr"] = prr(p)
    n_nan = int(p["prr"].isna().sum())
    if n_nan:
        # n_reac == a: event only ever reported with this drug → rank at the top
        p["prr"] = p["prr"].fillna(p["prr"].max())
        print(f"  PRR undefined for {n_nan:,} pairs (n_reac == a) → set to max finite PRR")
    p["ic"], p["ic025"] = bcpnn(p["a"], p["E"])
    p["ebgm"], p["eb05"] = ebgm(p["a"].values, p["E"].values, prior)
    return p


def main() -> None:
    print(f"Counting reports with year <= {FEATURE_CUTOFF_YEAR} …")
    counts, drugs = feature_counts()
    prior = fit_mgps_prior(counts["a"].values, counts["E"].values)
    print(f"  {len(counts):,} pairs; MGPS prior (a1,b1,a2,b2,p) = " + ", ".join(f"{v:.4g}" for v in prior))

    labels = pd.read_parquet(LABELS_SIDER, columns=LABEL_COLS)
    pairs = eligible_pairs(counts, labels)
    print(f"Eligible labeled pairs (>= {TASK_A_MIN_REPORTS} reports): {len(pairs):,}, "
          f"{pairs['label'].mean():.1%} positive, {pairs['ingredients'].nunique():,} ingredient sets")

    folds = assign_folds(pairs["ingredients"])
    pairs, n_dropped = pair_folds(pairs, folds)
    print(f"Ingredient-disjoint folds: {len(folds):,} ingredients; dropped {n_dropped:,} "
          f"combination pairs spanning folds; {len(pairs):,} pairs kept")

    q, bad = fold_quality(pairs)
    print("\nFold quality:")
    print(q.to_string(float_format=lambda v: f"{v:.3f}"))
    if bad:
        print("\nFold quality limits breached — stopping:\n  " + "\n  ".join(bad), file=sys.stderr)
        sys.exit(1)

    pairs = add_scores(pairs, prior)
    pairs = pairs.merge(drugs, on="drugname", how="left")
    pairs["n_ingredients"] = pairs["ingredients"].str.count(r"\|") + 1
    scores = ["ror", "prr", "ic", "ic025", "ebgm", "eb05"]
    assert np.isfinite(pairs[scores + DRUG_FEATURES].to_numpy(float)).all(), "non-finite scores/features"

    TASK_A_FOLDS.parent.mkdir(parents=True, exist_ok=True)
    folds.to_parquet(TASK_A_FOLDS, index=False)
    keep = PAIR + ["ingredients", "label", "fold", "a", "n_drug", "n_reac", "N", "E"] + scores + DRUG_FEATURES
    pairs[keep].to_parquet(TASK_A_PAIRS, index=False)
    pd.Series(prior, index=["a1", "b1", "a2", "b2", "p"]).to_json(TASK_A_PAIRS.with_suffix(".prior.json"))
    print(f"\nSaved → {TASK_A_PAIRS}, {TASK_A_FOLDS}")


if __name__ == "__main__":
    main()
