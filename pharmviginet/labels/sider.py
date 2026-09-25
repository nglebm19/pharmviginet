#!/usr/bin/env python3
"""
SIDER 4.1 → (ingredient, pt) known side-effect pairs.

SIDER is CC BY-NC-SA and contains MedDRA terms. Files are downloaded into
data/external/sider/ (gitignored) and never redistributed.

SIDER drug names are sometimes truncated ("gamma-aminobutyric"), so if RxNav
finds no ingredient for the SIDER name, the PubChem compound title is tried.

Usage:
    python -m pharmviginet.labels.sider
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from pharmviginet.config import EXTERNAL, SIDER_DIR, SIDER_PAIRS
from pharmviginet.labels.drug_norm import RxNormClient, normalize, normalize_names

SIDER_URL = "https://sideeffects.embl.de/media/download/"
FILES = ["meddra_all_se.tsv.gz", "drug_names.tsv"]
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/Title/JSON"


def download() -> None:
    SIDER_DIR.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        out = SIDER_DIR / f
        if out.exists():
            continue
        r = requests.get(SIDER_URL + f, timeout=120)
        r.raise_for_status()
        out.write_bytes(r.content)
        print(f"Downloaded {f} ({len(r.content) / 1e6:.1f} MB)")


def stitch_to_pubchem(stitch_flat: str) -> int:
    """CID1xxxxxxxx (flat STITCH id) → PubChem CID."""
    return int(stitch_flat[3:]) - 100_000_000


def load_side_effects() -> pd.DataFrame:
    se = pd.read_csv(SIDER_DIR / "meddra_all_se.tsv.gz", sep="\t", header=None,
                     names=["stitch_flat", "stitch_stereo", "umls_label",
                            "meddra_type", "umls_meddra", "se_name"])
    se = se[se["meddra_type"] == "PT"]
    se["pt"] = se["se_name"].str.strip().str.lower()
    return se[["stitch_flat", "pt"]].drop_duplicates()


def load_drug_names() -> pd.DataFrame:
    return pd.read_csv(SIDER_DIR / "drug_names.tsv", sep="\t", header=None,
                       names=["stitch_flat", "sider_name"])


def pubchem_title(cid: int, session: requests.Session) -> str | None:
    time.sleep(0.2)  # PubChem limit: 5 req/s
    try:
        r = session.get(PUBCHEM.format(cid=cid), timeout=30)
        if r.ok:
            return r.json()["PropertyTable"]["Properties"][0].get("Title")
    except (requests.RequestException, ValueError, KeyError, IndexError):
        pass
    return None


def map_sider_drugs(drugs: pd.DataFrame) -> pd.DataFrame:
    """stitch_flat → ingredients ('|'-joined), trying SIDER name then PubChem title."""
    client = RxNormClient()
    by_name = normalize_names(drugs["sider_name"], EXTERNAL / "rxnav_cache_sider.parquet", client)
    out = drugs.merge(by_name[["drugname", "ingredients"]],
                      left_on="sider_name", right_on="drugname", how="left").drop(columns="drugname")

    missing = out["ingredients"].fillna("") == ""
    if missing.any():
        print(f"{missing.sum()} SIDER drugs unmapped by name — trying PubChem titles")
        session = requests.Session()
        for idx in out.index[missing]:
            title = pubchem_title(stitch_to_pubchem(out.at[idx, "stitch_flat"]), session)
            if title:
                out.at[idx, "ingredients"] = normalize(title.upper(), client)["ingredients"]
    return out


def main() -> None:
    download()
    se = load_side_effects()
    drugs = map_sider_drugs(load_drug_names())
    print(f"SIDER drugs mapped: {(drugs['ingredients'].fillna('') != '').mean():.1%} of {len(drugs)}")

    # Only single-ingredient SIDER drugs, so a pair's ingredient is unambiguous
    drugs = drugs[(drugs["ingredients"].fillna("") != "") & ~drugs["ingredients"].str.contains("|", regex=False)]
    pairs = (se.merge(drugs[["stitch_flat", "ingredients"]], on="stitch_flat")
               .rename(columns={"ingredients": "ingredient"})[["ingredient", "pt"]]
               .drop_duplicates())
    pairs.to_parquet(SIDER_PAIRS, index=False)
    print(f"{len(pairs):,} pairs, {pairs['ingredient'].nunique()} ingredients, "
          f"{pairs['pt'].nunique()} PTs → {SIDER_PAIRS}")


if __name__ == "__main__":
    main()
