import pandas as pd

from pharmviginet.labels.build_labels import label_pairs
from pharmviginet.labels.drug_norm import clean_name, normalize_names
from pharmviginet.labels.sider import stitch_to_pubchem


def test_clean_name():
    assert clean_name("alatrofloxacin mesylate.") == "ALATROFLOXACIN MESYLATE"
    assert clean_name("ALFUZOSIN (ALFUZOSIN)") == "ALFUZOSIN"
    assert clean_name("ETHYLENEDIAMINE TETRAACETIC ACID (EDTA)") == "ETHYLENEDIAMINE TETRAACETIC ACID (EDTA)"
    assert clean_name("MM?398") == "MM 398"
    assert clean_name("  Humira  ") == "HUMIRA"


def test_stitch_to_pubchem():
    assert stitch_to_pubchem("CID100000085") == 85


def _fake_normalizer(calls):
    table = {"HUMIRA": "adalimumab", "AVONEX": "interferon beta-1a"}

    def fake(clean, client):
        calls.append(clean)
        return {"clean": clean, "rxcui": "1", "score": 1.0,
                "match_name": clean.lower(), "ingredients": table.get(clean, "")}
    return fake


def test_normalize_names_resumes_from_cache(tmp_path):
    cache = tmp_path / "cache.parquet"
    calls: list[str] = []
    df = normalize_names(["HUMIRA", "Humira.", "AVONEX"], cache,
                         client=object(), normalizer=_fake_normalizer(calls))
    assert calls == ["HUMIRA", "AVONEX"]            # "Humira." dedups to HUMIRA
    assert df.set_index("drugname").loc["Humira.", "ingredients"] == "adalimumab"

    calls.clear()
    normalize_names(["HUMIRA", "AVONEX", "ENBREL"], cache,
                    client=object(), normalizer=_fake_normalizer(calls))
    assert calls == ["ENBREL"]                      # cached names skipped


def test_label_pairs():
    sider = pd.DataFrame({"ingredient": ["a", "a", "b"],
                          "pt": ["nausea", "rash", "headache"]})
    drug_map = pd.DataFrame({
        "drugname": ["DRUG_A", "DRUG_AB", "DRUG_AX", "DRUG_X", "UNMAPPED"],
        "ingredients": ["a", "a|b", "a|x", "x", ""],
    })
    pairs = pd.DataFrame({
        "drugname": ["DRUG_A", "DRUG_A", "DRUG_A", "DRUG_AB", "DRUG_AX", "DRUG_AX", "DRUG_X", "UNMAPPED"],
        "pt":       ["Nausea", "Headache", "Fever", "Headache", "Rash", "Headache", "Nausea", "Nausea"],
    })
    got = {(r.drugname, r.pt): r.label for r in label_pairs(pairs, drug_map, sider).itertuples()}
    assert got == {
        ("DRUG_A", "Nausea"): 1,      # listed in SIDER
        ("DRUG_A", "Headache"): 0,    # drug + pt known to SIDER, pair not listed
        ("DRUG_AB", "Headache"): 1,   # combo: any ingredient listed → positive
        ("DRUG_AX", "Rash"): 1,       # positive even though x is not in SIDER
        # DRUG_A/Fever: pt unknown to SIDER → dropped
        # DRUG_AX/Headache: x not in SIDER → can't claim negative → dropped
        # DRUG_X, UNMAPPED → dropped
    }
