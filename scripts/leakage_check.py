"""Read-only check: SIDER label overlap between time-split train and test pairs.

Measures why the legacy time split (FB-D5) is not a valid Task A evaluation (FB-D11).
A SIDER label is a static function of (RxNorm ingredients, PT). If the same key is
already labeled in the train period, a lookup "model" can copy its label.

Test pairs follow `models.baseline.load_split(..., labels="sider")`: unique
(drugname, pt) rows of test.parquet with a non-null `ror`, inner-joined with
labels_sider. Train-period keys come from train.parquet joined the same way.

Run from the repository root:
    python -m scripts.leakage_check [--output /tmp/leakage.json]

Reads ml/train.parquet, ml/test.parquet and labels_sider.parquet in bounded batches;
writes nothing unless --output is given. Output holds aggregates only, no label rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pharmviginet import config as c
from pharmviginet.utils.metrics import stratified_auc

PAIR = ["drugname", "pt"]


def unique_pairs(path: Path) -> pd.DataFrame:
    """Unique (drugname, pt) pairs with a non-null ror, read one batch at a time."""
    parts = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2_000_000, columns=PAIR + ["ror"]):
        df = batch.to_pandas().dropna(subset=PAIR + ["ror"])
        parts.append(df[PAIR].astype(str).drop_duplicates())
    return pd.concat(parts, ignore_index=True).drop_duplicates()


def keyed_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """[drugname, pt, ingredients, label] → adds key = ingredients + '#' + lowercase PT.

    build_labels assigns one label per key; fail loudly if that ever changes.
    """
    out = pd.DataFrame(labels.loc[:, PAIR + ["ingredients", "label"]]).copy()
    out["key"] = out["ingredients"].astype(str) + "#" + out["pt"].astype(str).str.strip().str.lower()
    if out.groupby("key")["label"].nunique().max() > 1:
        raise ValueError("labels disagree within an (ingredients, PT) key")
    return out


def lookup_overlap(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """train/test: labeled pairs [drugname, pt, key, label].

    Scores each test pair by copying the train-period label of its key; unseen keys
    get the train positive rate. Returns overlap fraction and stratified AUCs.
    """
    lookup: dict[str, float] = (train.groupby("key")["label"].first().astype(float).to_dict())
    seen = test["key"].isin(list(lookup)).to_numpy()
    unseen_score = float(train["label"].to_numpy(float).mean())
    score = np.array([lookup.get(k, unseen_score) for k in test["key"]], dtype=float)
    y = test["label"].to_numpy(int)
    pt = test["pt"].to_numpy()
    auc_all, strata_all = stratified_auc(y, score, pt)
    auc_seen, strata_seen = stratified_auc(y[seen], score[seen], pt[seen])
    return {
        "test_pairs": int(len(test)),
        "train_labeled_keys": len(lookup),
        "test_keys_labeled_in_train_fraction": float(seen.mean()),
        "lookup_auc_strat": auc_all, "lookup_n_strata": strata_all,
        "lookup_auc_strat_seen_only": auc_seen, "lookup_n_strata_seen_only": strata_seen,
        "unseen_score": "train-period positive rate",
    }


def check() -> dict:
    labels = keyed_labels(pd.read_parquet(c.LABELS_SIDER, columns=PAIR + ["ingredients", "label"]))
    test = unique_pairs(c.TEST_PARQUET).merge(labels, on=PAIR)
    train = unique_pairs(c.TRAIN_PARQUET).merge(labels, on=PAIR)
    return {"split": {"train_max_year": c.TRAIN_MAX_YEAR, "test_min_year": c.TEST_MIN_YEAR},
            **lookup_overlap(train, test)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check()
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
