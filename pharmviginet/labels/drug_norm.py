#!/usr/bin/env python3
"""
Drug name normalization — map free-text FAERS drug names to RxNorm ingredients.

FAERS `drugname` is verbatim reporter text ("GABAPENTIN TABLETS", "HUMIRA",
"ALFUZOSIN (ALFUZOSIN)"). SIDER is keyed by generic compound. Both sides are
mapped to RxNorm ingredient names (tty=IN) via the NLM RxNav API so they join.

Results are cached to parquet and the run is resumable.

Usage:
    python -m pharmviginet.labels.drug_norm                 # names with >= 10 rows
    python -m pharmviginet.labels.drug_norm --min-rows 50   # smaller, faster run
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

from pharmviginet.config import EXTERNAL, MASTER_PARQUET, RXNORM_MAP

RXNAV = "https://rxnav.nlm.nih.gov/REST"
MAX_RPS = 15          # RxNav allows 20 req/s per IP; stay under it
FLUSH_EVERY = 500

_PAREN_RE = re.compile(r"\s*\(([^()]*)\)\s*")
_SPACE_RE = re.compile(r"\s+")


def clean_name(name: str) -> str:
    """Conservative cleanup. Leaves dose/form text alone — RxNav handles it."""
    s = str(name).upper().strip()
    s = s.rstrip(".,;: ")
    # "ALFUZOSIN (ALFUZOSIN)" → "ALFUZOSIN"; keep informative parentheses
    m = _PAREN_RE.search(s)
    if m and m.group(1).strip() == _PAREN_RE.sub(" ", s).strip():
        s = m.group(1)
    s = s.replace("?", " ")
    return _SPACE_RE.sub(" ", s).strip()


class RxNormClient:
    """Thin rate-limited RxNav client with retries."""

    def __init__(self, session: Optional[requests.Session] = None, max_rps: float = MAX_RPS):
        self.session = session or requests.Session()
        self.min_interval = 1.0 / max_rps
        self._last = 0.0
        self._ing_cache: dict[str, list[str]] = {}

    def _get(self, path: str, params: dict) -> dict:
        for attempt in range(5):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.session.get(f"{RXNAV}{path}", params=params, timeout=30)
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError):
                time.sleep(2 ** attempt)
        raise RuntimeError(f"RxNav failed after retries: {path} {params}")

    def approximate(self, term: str) -> tuple[Optional[str], Optional[float], Optional[str]]:
        """Best RxNorm match → (rxcui, score, matched name)."""
        d = self._get("/approximateTerm.json", {"term": term, "maxEntries": 5})
        cands = d.get("approximateGroup", {}).get("candidate", []) or []
        if not cands:
            return None, None, None
        # Prefer a candidate from the RXNORM source (has a readable name)
        best = next((c for c in cands if c.get("source") == "RXNORM"), cands[0])
        return best.get("rxcui"), float(best.get("score", 0) or 0), best.get("name")

    def ingredients(self, rxcui: str) -> list[str]:
        """rxcui → sorted list of lowercase ingredient names (tty=IN)."""
        if rxcui in self._ing_cache:
            return self._ing_cache[rxcui]
        d = self._get(f"/rxcui/{rxcui}/related.json", {"tty": "IN"})
        names = set()
        for g in d.get("relatedGroup", {}).get("conceptGroup", []) or []:
            for c in g.get("conceptProperties", []) or []:
                names.add(c["name"].lower())
        out = sorted(names)
        self._ing_cache[rxcui] = out
        return out


def normalize(clean: str, client: RxNormClient) -> dict:
    rxcui, score, match = client.approximate(clean)
    ings = client.ingredients(rxcui) if rxcui else []
    return {"clean": clean, "rxcui": rxcui, "score": score,
            "match_name": match, "ingredients": "|".join(ings)}


def normalize_names(names: Iterable[str], cache_path: Path,
                    client: Optional[RxNormClient] = None,
                    normalizer: Callable[[str, RxNormClient], dict] = normalize) -> pd.DataFrame:
    """
    Map raw names → ingredients. Resumable: cleaned names already in
    `cache_path` are skipped. Returns one row per raw name.
    """
    client = client or RxNormClient()
    raw = pd.DataFrame({"drugname": pd.unique(pd.Series(list(names), dtype=object))})
    raw["clean"] = raw["drugname"].map(clean_name)

    done = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame(
        columns=["clean", "rxcui", "score", "match_name", "ingredients"])
    seen = set(done["clean"])
    todo = [c for c in pd.unique(raw["clean"]) if c and c not in seen]

    rows: list[dict] = []

    def flush():
        nonlocal done, rows
        if rows:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            done.to_parquet(cache_path, index=False)
            rows = []

    for i, c in enumerate(tqdm(todo, desc="RxNav", unit="name"), 1):
        rows.append(normalizer(c, client))
        if i % FLUSH_EVERY == 0:
            flush()
    flush()

    return raw.merge(done, on="clean", how="left")


def faers_drugnames(min_rows: int) -> list[str]:
    col = pq.read_table(MASTER_PARQUET, columns=["drugname"])["drugname"]
    return [v["values"] for v in pc.value_counts(col).to_pylist()
            if v["values"] is not None and v["counts"] >= min_rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-rows", type=int, default=10,
                    help="only map drug names with at least this many FAERS rows")
    args = ap.parse_args()

    names = faers_drugnames(args.min_rows)
    print(f"{len(names):,} FAERS drug names with >= {args.min_rows} rows")
    df = normalize_names(names, EXTERNAL / "rxnav_cache_faers.parquet")
    df.to_parquet(RXNORM_MAP, index=False)
    mapped = (df["ingredients"].fillna("") != "").mean()
    print(f"Mapped to >=1 ingredient: {mapped:.1%} of names → {RXNORM_MAP}")


if __name__ == "__main__":
    main()
