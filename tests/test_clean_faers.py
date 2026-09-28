import numpy as np
import pandas as pd

import clean_faers as cf
from pharmviginet.labels.build_labels import label_pairs


def _load(tmp_path, text, table):
    path = tmp_path / "t.txt"
    path.write_text(text, encoding="latin1")
    return cf.normalise_table(cf.load_table_file(path), table)


def test_legacy_trailing_delimiter_does_not_shift_columns(tmp_path):
    # legacy AERS: header has no trailing "$", data rows do
    df = _load(tmp_path, "ISR$PT\n4204616$ABDOMINAL PAIN$\n4204617$NAUSEA$\n", "REAC")
    assert list(df.columns) == ["primaryid", "pt"]
    assert df["primaryid"].tolist() == [4204616, 4204617]
    assert df["pt"].tolist() == ["ABDOMINAL PAIN", "NAUSEA"]


def test_modern_file_reads_unchanged(tmp_path):
    df = _load(tmp_path, "primaryid$caseid$pt\n34483284$3448328$Laryngomalacia\n", "REAC")
    assert df.loc[0, "primaryid"] == 34483284 and df.loc[0, "caseid"] == 3448328
    assert df.loc[0, "pt"] == "LARYNGOMALACIA"


def test_uppercase_legacy_headers_are_renamed(tmp_path):
    demo = _load(tmp_path, "ISR$CASE$I_F_COD$FOLL_SEQ$IMAGE$FDA_DT\n1$10$I$$1-1$20040101$\n", "DEMO")
    assert {"primaryid", "caseid", "fda_dt"} <= set(demo.columns)
    assert not {"i_f_cod", "foll_seq", "image"} & set(demo.columns)
    assert (demo.loc[0, "primaryid"], demo.loc[0, "caseid"]) == (1, 10)
    indi = _load(tmp_path, "ISR$DRUG_SEQ$INDI_PT\n1$5$Hypertension$\n", "INDI")
    assert "indi_drug_seq" in indi.columns and indi.loc[0, "indi_pt"] == "HYPERTENSION"


def test_ids_share_one_integer_dtype_across_eras(tmp_path):
    legacy = _load(tmp_path, "ISR$PT\n4204616$NAUSEA$\n", "REAC")
    modern = _load(tmp_path, "primaryid$caseid$pt\n34483284$3448328$Nausea\n", "REAC")
    assert legacy["primaryid"].dtype == modern["primaryid"].dtype == "Int64"
    both = pd.concat([legacy, modern], ignore_index=True)
    assert both["primaryid"].astype(str).tolist() == ["4204616", "34483284"]


def test_key_coverage_flags_null_keys():
    df = pd.DataFrame({"primaryid": pd.array([1, None], dtype="Int64"),
                       "caseid": pd.array([1, 2], dtype="Int64")})
    cov = cf.key_coverage(df, "DEMO")
    assert cov["primaryid"] == 0.5 < cf.MIN_COVERAGE["primaryid"]
    assert cov["caseid"] == 1.0


def _demo(rows):
    cols = ["primaryid", "caseid", "caseversion", "fda_dt", "year", "era", "age"]
    return pd.DataFrame(rows, columns=cols)


def test_dedup_demo_priority_and_whole_rows():
    df = _demo([
        # legacy case 10: two ISRs, later fda_dt wins; its null age must not be back-filled
        (4200001, 10, np.nan, 20040101, 2004, "aers", 50.0),
        (4200002, 10, np.nan, 20040601, 2004, "aers", np.nan),
        # legacy case 11: same fda_dt, higher ISR wins
        (4200003, 11, np.nan, 20050101, 2005, "aers", 1.0),
        (4200004, 11, np.nan, 20050101, 2005, "aers", 2.0),
        # case 12 continues into FAERS: modern version wins over later-looking legacy row
        (4200005, 12, np.nan, 20990101, 2011, "aers", 3.0),
        (120001, 12, 1, 20130101, 2013, "faers", 4.0),
        (120002, 12, 2, 20120101, 2013, "faers", 5.0),
        # legacy row with no case → dropped
        (4200006, None, np.nan, 20060101, 2006, "aers", 6.0),
    ])
    df["caseid"] = df["caseid"].astype("Int64")
    out = cf.dedup_demo(df).set_index("caseid")
    assert out.loc[10, "primaryid"] == 4200002 and np.isnan(out.loc[10, "age"])
    assert out.loc[11, "primaryid"] == 4200004
    assert out.loc[12, "primaryid"] == 120002       # highest caseversion beats fda_dt
    assert len(out) == 3


def test_check_year_coverage_detects_lost_year():
    before = pd.DataFrame({"year": [2004, 2004, 2013]})
    assert cf.check_year_coverage(before, pd.DataFrame({"year": [2013]})) == ["2004: 2 rows → 0 cases"]
    assert cf.check_year_coverage(before, pd.DataFrame({"year": [2004, 2013]})) == []


def test_pt_case_variants_map_to_same_sider_event(tmp_path):
    legacy = _load(tmp_path, "ISR$PT\n1$NAUSEA$\n", "REAC")
    modern = _load(tmp_path, "primaryid$caseid$pt\n2$2$Nausea\n", "REAC")
    assert legacy.loc[0, "pt"] == modern.loc[0, "pt"]
    # both spellings still hit the lowercase SIDER event at the label join
    pairs = pd.DataFrame({"drugname": ["D", "D2"], "pt": ["NAUSEA", "Nausea"], "n_reports": [3, 3]})
    drug_map = pd.DataFrame({"drugname": ["D", "D2"], "ingredients": ["x", "x"]})
    sider = pd.DataFrame({"ingredient": ["x"], "pt": ["nausea"]})
    out = label_pairs(pairs, drug_map, sider)
    assert out["label"].tolist() == [1, 1]
