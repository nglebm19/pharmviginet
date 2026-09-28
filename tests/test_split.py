import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import clean_faers as cf
from pharmviginet import config
from pharmviginet.models import baseline

BOUNDS = {"train": (0, 2022), "val": (2023, 2023), "test": (2024, 9999)}


def _master(tmp_path, years=(2020, 2021, 2022, 2023, 2024, 2025), per_year=3):
    ys = [y for y in years for _ in range(per_year)]
    df = pd.DataFrame({"primaryid": pd.array(range(len(ys)), dtype="Int64"),
                       "drugname": ["D"] * len(ys), "pt": ["P"] * len(ys),
                       "year": ys, "label": [i % 2 for i in range(len(ys))]})
    path = tmp_path / "master.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, row_group_size=4)
    return path, df


def _paths(tmp_path):
    d = tmp_path / "ml"
    return {k: d / f"{k}.parquet" for k in BOUNDS}, d / "split_manifest.json"


def test_splits_partition_master_by_year(tmp_path):
    src, df = _master(tmp_path)
    paths, manifest = _paths(tmp_path)
    info = cf.write_splits(src, paths, BOUNDS, manifest)
    got = {k: pq.read_table(p).to_pandas() for k, p in paths.items()}
    assert got["train"]["year"].max() == 2022 and got["val"]["year"].unique().tolist() == [2023]
    assert got["test"]["year"].min() == 2024
    ids = sorted(pd.concat(got.values())["primaryid"].tolist())
    assert ids == sorted(df["primaryid"].tolist())                 # every row exactly once
    # order preserved within each split
    assert got["train"]["primaryid"].tolist() == df[df.year <= 2022]["primaryid"].tolist()
    assert {k: v["rows"] for k, v in info["splits"].items()} == {"train": 9, "val": 3, "test": 6}
    assert cf.check_split(paths, manifest, src) == []


def test_schema_unchanged_and_deterministic(tmp_path):
    src, _ = _master(tmp_path)
    paths, manifest = _paths(tmp_path)
    cf.write_splits(src, paths, BOUNDS, manifest)
    first = {k: pq.read_table(p) for k, p in paths.items()}
    cf.write_splits(src, paths, BOUNDS, manifest)
    for k, p in paths.items():
        t = pq.read_table(p)
        assert t.schema.equals(pq.ParquetFile(src).schema_arrow)
        assert t.equals(first[k])
        assert "ror_train" not in t.column_names


def test_single_pass_over_row_groups_in_order(tmp_path, monkeypatch):
    src, _ = _master(tmp_path)                                     # 18 rows → 5 row groups
    paths, manifest = _paths(tmp_path)
    read = []
    orig = pq.ParquetFile.read_row_group
    monkeypatch.setattr(pq.ParquetFile, "read_row_group",
                        lambda self, i, *a, **k: (read.append(i), orig(self, i, *a, **k))[1])
    cf.write_splits(src, paths, BOUNDS, manifest)
    assert read == list(range(pq.ParquetFile(src).num_row_groups))


def test_bounds_with_a_gap_fail(tmp_path):
    src, _ = _master(tmp_path)
    paths, manifest = _paths(tmp_path)
    gap = {"train": (0, 2021), "val": (2023, 2023), "test": (2024, 9999)}   # 2022 uncovered
    with pytest.raises(ValueError, match="!= master rows"):
        cf.write_splits(src, paths, gap, manifest)
    assert not manifest.exists()


def test_failed_check_publishes_nothing(tmp_path):
    src, _ = _master(tmp_path, years=(2020, 2021))                 # no val/test years
    paths, manifest = _paths(tmp_path)
    paths["train"].parent.mkdir()
    paths["train"].write_bytes(b"old")
    manifest.write_text("old")
    with pytest.raises(ValueError, match="no rows"):
        cf.write_splits(src, paths, BOUNDS, manifest)
    assert paths["train"].read_bytes() == b"old" and manifest.read_text() == "old"
    assert not list(paths["train"].parent.glob(".tmp-*"))


def test_manifest_is_the_completion_marker(tmp_path):
    src, _ = _master(tmp_path)
    paths, manifest = _paths(tmp_path)
    cf.write_splits(src, paths, BOUNDS, manifest)
    info = json.loads(manifest.read_text())
    assert info["source_rows"] == 18 and [c for c, _ in info["schema"]][:2] == ["primaryid", "drugname"]
    assert info["splits"]["val"]["rows_by_year"] == {"2023": 3}
    manifest.unlink()
    assert "missing" in cf.check_split(paths, manifest)[0]
    cf.write_splits(src, paths, BOUNDS, manifest)
    pq.write_table(pq.read_table(paths["val"]).slice(0, 1), paths["val"])   # tamper
    assert any("val" in p for p in cf.check_split(paths, manifest))


def test_split_bounds_follow_config():
    b = cf.split_bounds()
    assert b["train"][1] == config.TRAIN_MAX_YEAR == config.FEATURE_CUTOFF_YEAR
    assert b["val"][0] == b["train"][1] + 1 and b["test"][0] == b["val"][1] + 1


def test_train_counts_computes_ror_train(tmp_path, monkeypatch):
    rows = [(i, "D", "E") for i in (1, 2)] + [(i, "D", "O") for i in (3, 4)]
    rows += [(5, "X", "E")] + [(i, "X", "O") for i in range(6, 11)]
    path = tmp_path / "train.parquet"
    pd.DataFrame(rows, columns=["primaryid", "drugname", "pt"]).to_parquet(path)
    monkeypatch.setattr(baseline, "TRAIN_PARQUET", path)
    c, _ = baseline.train_counts()
    r = c.set_index(["drugname", "pt"]).loc[("D", "E"), "ror_train"]
    # a=2, b=2, c=1, d=5 → 2·5 / (2.5·1.5)
    assert r == pytest.approx(10 / 3.75)
