"""scripts/leakage_check.py on synthetic data (no local FAERS/SIDER files needed)."""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import leakage_check as lc


def _labels(n_drugs=30):
    # drug d<i> has ingredient i<i>; label differs by drug so each PT has both classes
    rows = [(f"D{i}", pt, f"i{i}", int((i + j) % 3 == 0))
            for i in range(n_drugs) for j, pt in enumerate(["NAUSEA", "RASH"])]
    return pd.DataFrame.from_records(rows, columns=pd.Index(["drugname", "pt", "ingredients", "label"]))


def test_keyed_labels_rejects_conflicting_labels():
    labels = pd.DataFrame({"drugname": ["A", "B"], "pt": ["RASH", "rash "],
                           "ingredients": ["x", "x"], "label": [1, 0]})
    with pytest.raises(ValueError):
        lc.keyed_labels(labels)


def test_lookup_overlap_keys_on_ingredients_not_drugname():
    labels = lc.keyed_labels(_labels())
    train = pd.DataFrame(labels.loc[labels.drugname.isin([f"D{i}" for i in range(24)])])
    test = labels.copy()
    # a brand name for ingredient i0 is unseen as a drugname but shares the key
    test.loc[test.drugname == "D0", "drugname"] = "BRAND0"
    out = lc.lookup_overlap(train, test)
    assert out["test_pairs"] == 60
    assert out["train_labeled_keys"] == 48
    assert out["test_keys_labeled_in_train_fraction"] == pytest.approx(48 / 60)
    assert out["lookup_auc_strat_seen_only"] == pytest.approx(1.0)
    assert out["lookup_n_strata_seen_only"] == 2
    assert 0.5 < out["lookup_auc_strat"] < 1.0


def test_unique_pairs_drops_null_ror_and_duplicates(tmp_path):
    path = tmp_path / "split.parquet"
    pq.write_table(pa.table({"drugname": ["A", "A", "B", None], "pt": ["X", "X", "Y", "X"],
                             "ror": [1.0, 2.0, None, 1.0]}), path)
    out = lc.unique_pairs(path)
    assert out.values.tolist() == [["A", "X"]]
