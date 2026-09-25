#!/usr/bin/env python3
"""
Build independent (non-ROR) labels for FAERS drug–event pairs from SIDER.

For each FAERS (drugname, pt) pair with >= 3 reports:
    label = 1  if SIDER lists (ingredient, pt) for any of the drug's ingredients
    label = 0  if every ingredient is in SIDER and the pt is in SIDER's
               vocabulary, but no ingredient lists it (closed-world assumption —
               adds some label noise, documented in the benchmark)
    dropped    otherwise (drug unmapped / not in SIDER, or pt unknown to SIDER)

Needs: data/external/rxnorm_map.parquet (drug_norm) and sider_pairs.parquet (sider).

Usage:
    python -m pharmviginet.labels.build_labels
"""
from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq

from pharmviginet.config import LABELS_SIDER, MASTER_PARQUET, RXNORM_MAP, SIDER_PAIRS

MIN_REPORTS = 3


def faers_pairs() -> pd.DataFrame:
    tbl = pq.read_table(MASTER_PARQUET, columns=["drugname", "pt", "n_reports"])
    df = tbl.to_pandas()
    df = df.dropna(subset=["drugname", "pt"])
    pairs = df.groupby(["drugname", "pt"], observed=True)["n_reports"].max().reset_index()
    return pairs[pairs["n_reports"] >= MIN_REPORTS]


def label_pairs(pairs: pd.DataFrame, drug_map: pd.DataFrame, sider: pd.DataFrame) -> pd.DataFrame:
    """
    pairs:    [drugname, pt, ...]
    drug_map: [drugname, ingredients]  ('|'-joined, '' if unmapped)
    sider:    [ingredient, pt]         (pt lowercase)
    → pairs with added [ingredients, label], unlabeled pairs removed.
    """
    sider_drugs = set(sider["ingredient"])
    sider_pts = set(sider["pt"])

    df = pairs.merge(drug_map[["drugname", "ingredients"]], on="drugname", how="left")
    df["ingredients"] = df["ingredients"].fillna("")
    df["pt_l"] = df["pt"].str.strip().str.lower()
    df = df[(df["ingredients"] != "") & df["pt_l"].isin(sider_pts)]

    ex = (df[["ingredients", "pt_l"]].drop_duplicates()
          .assign(ingredient=lambda x: x["ingredients"].str.split("|"))
          .explode("ingredient"))
    ex["in_sider_drug"] = ex["ingredient"].isin(sider_drugs)
    ex = ex.merge(sider.rename(columns={"pt": "pt_l"}).assign(known=True),
                  on=["ingredient", "pt_l"], how="left")
    ex["known"] = ex["known"].eq(True)

    agg = ex.groupby(["ingredients", "pt_l"]).agg(
        any_known=("known", "any"), all_in_sider=("in_sider_drug", "all")).reset_index()
    agg["label"] = pd.NA
    agg.loc[agg["all_in_sider"], "label"] = 0
    agg.loc[agg["any_known"], "label"] = 1
    agg = agg.dropna(subset=["label"])

    out = df.merge(agg[["ingredients", "pt_l", "label"]], on=["ingredients", "pt_l"])
    out["label"] = out["label"].astype("int8")
    return out.drop(columns="pt_l")


def main() -> None:
    pairs = faers_pairs()
    print(f"{len(pairs):,} FAERS (drugname, pt) pairs with >= {MIN_REPORTS} reports")
    labels = label_pairs(pairs, pd.read_parquet(RXNORM_MAP), pd.read_parquet(SIDER_PAIRS))
    labels["source"] = "sider4.1"
    labels.to_parquet(LABELS_SIDER, index=False)
    print(f"Labeled {len(labels):,} pairs ({len(labels) / len(pairs):.1%}), "
          f"positive rate {labels['label'].mean():.1%}, "
          f"{labels['drugname'].nunique():,} drug names → {LABELS_SIDER}")


if __name__ == "__main__":
    main()
