#!/usr/bin/env python3
"""
clean_faers.py — PharmVigiNet pipeline stages 2–5

Usage:
    python clean_faers.py --stage clean     # stage 2: normalize + concat → parquet
    python clean_faers.py --stage dedup     # stage 3: dedup cases, PS-filter drugs
    python clean_faers.py --stage smiles    # stage 4: PubChem SMILES lookup
    python clean_faers.py --stage join      # stage 5: master.parquet
    python clean_faers.py --stage split     # stage 6: ml/{train,val,test}.parquet, streamed
    python clean_faers.py --all             # run stages 2→3→4→5→6 in sequence
    python clean_faers.py --stage clean --table DEMO   # single table (debug)
"""
from __future__ import annotations

import argparse
import json
import os
import time
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

from pharmviginet.config import (
    MASTER_PARQUET, SPLIT_MANIFEST, TEST_MIN_YEAR, TEST_PARQUET, TRAIN_MAX_YEAR, TRAIN_PARQUET,
    VAL_PARQUET, VAL_YEARS,
)

# ── paths ──────────────────────────────────────────────────────────────────────
EXTRACT_DIR  = Path("data/extracted")
PROCESSED    = Path("data/processed")
LOG_DIR      = Path("data/logs")
AUDIT_REPORT = LOG_DIR / "audit_report.json"

# ── era detection ──────────────────────────────────────────────────────────────
def is_aers(year: int, q: int) -> bool:
    return year < 2012 or (year == 2012 and q <= 3)

# ── file finder (mirrors audit_faers.py logic) ─────────────────────────────────
def _ascii_dir(qdir: Path) -> Path:
    for name in ("ascii", "ASCII"):
        p = qdir / name
        if p.is_dir():
            return p
    return qdir

def find_table_file(qdir: Path, table: str, year: int, q: int) -> Optional[Path]:
    adir = _ascii_dir(qdir)
    if not adir.exists():
        return None
    yy = str(year)[2:]
    prefix = f"{table}{yy}Q{q}".upper()
    for p in adir.iterdir():
        if p.is_file() and p.stem.upper().startswith(prefix) and p.suffix.lower() == ".txt":
            return p
    return None

def parse_quarter(label: str) -> tuple[int, int]:
    """'2016Q1' → (2016, 1)"""
    year = int(label[:4])
    q = int(label[5])
    return year, q

def list_quarters() -> list[str]:
    quarters = []
    for qdir in sorted(EXTRACT_DIR.iterdir()):
        if qdir.is_dir() and len(qdir.name) == 6 and qdir.name[4] == "Q":
            quarters.append(qdir.name)
    return quarters

# ── schema normalisation maps (from audit schema diff) ─────────────────────────
# All tables: isr → primaryid
# DEMO: case → caseid
# INDI: drug_seq → indi_drug_seq
# THER: drug_seq → dsg_drug_seq
RENAME_MAP: dict[str, dict[str, str]] = {
    "DEMO": {"isr": "primaryid", "case": "caseid"},
    "DRUG": {"isr": "primaryid"},
    "REAC": {"isr": "primaryid"},
    "INDI": {"isr": "primaryid", "drug_seq": "indi_drug_seq"},
    "OUTC": {"isr": "primaryid"},
    "RPSR": {"isr": "primaryid"},
    "THER": {"isr": "primaryid", "drug_seq": "dsg_drug_seq"},
}

# AERS-only cols not present in FAERS — drop to unify schema
AERS_DROP = {"confid", "death_dt", "foll_seq", "i_f_cod", "image"}

ALL_TABLES = ["DEMO", "DRUG", "REAC", "INDI", "OUTC", "RPSR", "THER"]


def load_table_file(path: Path) -> pd.DataFrame:
    """Load single FAERS/AERS table file with correct params."""
    # index_col=False: legacy AERS data rows end with a trailing "$" that the header
    # lacks; without it pandas silently turns the first column (ISR) into the index
    # and shifts every other column one place left.
    df = pd.read_csv(path, sep="$", encoding="latin1", low_memory=False, on_bad_lines="skip",
                     index_col=False)
    # drop phantom trailing-$ column
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    # strip BOM artifact on column names (ï»¿primaryid → primaryid); legacy headers are
    # uppercase (ISR, CASE, PT), modern ones lowercase
    df.columns = [c.encode("latin1").decode("utf-8-sig").strip().lower() if isinstance(c, str) else c
                  for c in df.columns]
    return df


def normalise_table(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Rename AERS columns to FAERS equivalents and drop AERS-only cols."""
    renames = RENAME_MAP.get(table, {})
    # only rename cols that actually exist
    actual_renames = {k: v for k, v in renames.items() if k in df.columns}
    if actual_renames:
        df = df.rename(columns=actual_renames)
    # drop AERS-only cols
    drop_cols = [c for c in df.columns if c.lower() in AERS_DROP]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    # same integer dtype for IDs in every table, so string keys match in joins
    for col in ("primaryid", "caseid"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    # MedDRA terms: legacy files are uppercase, modern mixed case
    for col in ("pt", "indi_pt"):
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper()
    return df


def key_coverage(df: pd.DataFrame, table: str) -> dict[str, float]:
    """Share of non-null join keys: primaryid always, caseid for DEMO."""
    keys = ["primaryid"] + (["caseid"] if table == "DEMO" else [])
    return {k: float(df[k].notna().mean()) if k in df.columns else 0.0 for k in keys}


# DEMO caseid may be missing on a handful of legacy rows (7 in 2004–2012Q3)
MIN_COVERAGE = {"primaryid": 1.0, "caseid": 0.999}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — clean
# ─────────────────────────────────────────────────────────────────────────────
def stage_clean(tables: Optional[list[str]] = None) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    quarters = list_quarters()
    if not quarters:
        print(f"[clean] No quarters found in {EXTRACT_DIR}", file=sys.stderr)
        sys.exit(1)

    target_tables = tables or ALL_TABLES
    print(f"[clean] {len(quarters)} quarters × {len(target_tables)} tables")

    bad_keys: list[str] = []
    for table in target_tables:
        chunks: list[pd.DataFrame] = []
        missing = 0
        for qlabel in tqdm(quarters, desc=f"  {table}", ncols=80):
            year, q = parse_quarter(qlabel)
            qdir = EXTRACT_DIR / qlabel
            path = find_table_file(qdir, table, year, q)
            if path is None:
                missing += 1
                continue
            try:
                df = load_table_file(path)
                df = normalise_table(df, table)
                df["era"] = "aers" if is_aers(year, q) else "faers"
                df["quarter"] = qlabel
                df["year"] = year
                chunks.append(df)
            except Exception as e:
                print(f"[clean] WARN {qlabel}/{table}: {e}", file=sys.stderr)
                continue
            # checked outside the try so a failure cannot be swallowed as a WARN
            for k, v in key_coverage(df, table).items():
                if v < MIN_COVERAGE[k]:
                    bad_keys.append(f"{qlabel}/{table}: {k} non-null {v:.4f}")

        if not chunks:
            print(f"[clean] SKIP {table} — no files found", file=sys.stderr)
            continue

        out = pd.concat(chunks, ignore_index=True)
        # FAERS columns have mixed types across eras — cast object cols to str
        for col in out.columns[out.dtypes == object]:
            out[col] = out[col].astype(str)
        out_path = PROCESSED / f"{table.lower()}.parquet"
        out.to_parquet(out_path, index=False)
        print(f"[clean] {table}: {len(out):,} rows → {out_path} "
              f"({missing} quarters missing)")

    if bad_keys:
        print("[clean] ERROR: join keys missing —\n  " + "\n  ".join(bad_keys), file=sys.stderr)
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — dedup
# ─────────────────────────────────────────────────────────────────────────────
def dedup_demo(demo: pd.DataFrame) -> pd.DataFrame:
    """
    One row per caseid, kept whole (fields are never mixed across versions).
    Priority: highest caseversion, then latest fda_dt, then highest primaryid (ISR)
    as a deterministic tie-breaker. Legacy AERS rows have no caseversion and get 0,
    so a case that continues into FAERS keeps its FAERS version.
    """
    # modifies caseversion / fda_dt in place (DEMO is large; avoid a full copy)
    demo["caseversion"] = pd.to_numeric(demo["caseversion"], errors="coerce").fillna(0)
    no_case = demo["caseid"].isna()
    if no_case.any():
        print(f"[dedup] dropping {no_case.sum():,} DEMO rows with no caseid", file=sys.stderr)
        demo = demo[~no_case]
    demo["fda_dt"] = pd.to_numeric(demo["fda_dt"], errors="coerce")
    return (demo
            .sort_values(["caseversion", "fda_dt", "primaryid"], kind="stable", na_position="first")
            .drop_duplicates("caseid", keep="last"))


def check_year_coverage(before: pd.DataFrame, after: pd.DataFrame) -> list[str]:
    """Years that have DEMO rows but no case left after dedup."""
    b = before.groupby("year").size()
    a = after.groupby("year").size().reindex(b.index, fill_value=0)
    return [f"{y}: {b[y]:,} rows → 0 cases" for y in b.index if a[y] == 0]


def stage_dedup() -> None:
    # --- DEMO dedup: keep latest version per caseid ---
    demo_path = PROCESSED / "demo.parquet"
    if not demo_path.exists():
        print("[dedup] demo.parquet missing — run --stage clean first", file=sys.stderr)
        sys.exit(1)

    print("[dedup] Loading demo.parquet …")
    demo = pd.read_parquet(demo_path)
    before = len(demo)
    years_before = demo[["year"]]

    demo = dedup_demo(demo)
    empty = check_year_coverage(years_before, demo)
    if empty:
        print("[dedup] ERROR: years lost in dedup —\n  " + "\n  ".join(empty), file=sys.stderr)
        sys.exit(1)
    print("[dedup] cases per year: " + ", ".join(
        f"{y}: {n:,}" for y, n in demo.groupby("year").size().items()))

    after = len(demo)
    out_path = PROCESSED / "cases_deduped.parquet"
    demo.to_parquet(out_path, index=False)
    print(f"[dedup] DEMO: {before:,} → {after:,} rows ({before-after:,} dupes removed) → {out_path}")

    # --- DRUG: primary suspect filter + restrict to valid case IDs ---
    drug_path = PROCESSED / "drug.parquet"
    if not drug_path.exists():
        print("[dedup] drug.parquet missing — run --stage clean first", file=sys.stderr)
        sys.exit(1)

    print("[dedup] Loading drug.parquet …")
    drug = pd.read_parquet(drug_path)
    before_drug = len(drug)

    if "role_cod" in drug.columns:
        drug = drug[drug["role_cod"] == "PS"]
    else:
        print("[dedup] WARN: role_cod column missing from drug.parquet", file=sys.stderr)

    valid_ids = set(demo["primaryid"].astype(str))
    drug = drug[drug["primaryid"].astype(str).isin(valid_ids)]

    out_path_drug = PROCESSED / "drug_ps.parquet"
    drug.to_parquet(out_path_drug, index=False)
    print(f"[dedup] DRUG: {before_drug:,} → {len(drug):,} rows (PS only, valid cases) → {out_path_drug}")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — smiles
# ─────────────────────────────────────────────────────────────────────────────
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/IsomericSMILES/JSON"
SMILES_SAVE_INTERVAL = 500


def fetch_smiles(name: str) -> Optional[str]:
    try:
        r = requests.get(PUBCHEM_URL.format(requests.utils.quote(name)), timeout=10)
        if r.status_code == 200:
            props = r.json()["PropertyTable"]["Properties"][0]
            return (props.get("IsomericSMILES")
                    or props.get("CanonicalSMILES")
                    or props.get("SMILES"))
    except Exception:
        pass
    return None


def stage_smiles() -> None:
    drug_path = PROCESSED / "drug_ps.parquet"
    if not drug_path.exists():
        print("[smiles] drug_ps.parquet missing — run --stage dedup first", file=sys.stderr)
        sys.exit(1)

    drug = pd.read_parquet(drug_path, columns=["drugname"])
    all_names = drug["drugname"].dropna().str.upper().unique().tolist()
    print(f"[smiles] {len(all_names):,} unique drug names")

    out_path = PROCESSED / "drug_smiles_map.parquet"
    # resume: load already-fetched entries
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        done = set(existing["drugname"].str.upper())
        records = existing.to_dict("records")
        print(f"[smiles] Resuming — {len(done):,} already fetched")
    else:
        done = set()
        records = []

    todo = [n for n in all_names if n not in done]
    print(f"[smiles] {len(todo):,} remaining (~{len(todo)*0.2/3600:.1f} hrs at 5/sec)")

    for i, name in enumerate(tqdm(todo, desc="  PubChem", ncols=80), 1):
        smiles = fetch_smiles(name)
        records.append({"drugname": name, "smiles": smiles if smiles else "[UNK-MOL]"})
        time.sleep(0.2)
        if i % SMILES_SAVE_INTERVAL == 0:
            pd.DataFrame(records).to_parquet(out_path, index=False)

    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)
    unk = (df["smiles"] == "[UNK-MOL]").sum()
    print(f"[smiles] Done: {len(df):,} entries, {unk:,} UNK-MOL ({unk/len(df):.1%}) → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — join → master.parquet
# ─────────────────────────────────────────────────────────────────────────────
def compute_ror(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ROR for each (drugname, pt) pair.
    Returns df with columns: drugname, pt, n_reports, ror, ror_lower_ci, label
    """
    import numpy as np

    N = df[["primaryid"]].drop_duplicates().shape[0]
    pair_counts = (df
                   .groupby(["drugname", "pt"])["primaryid"]
                   .nunique()
                   .reset_index(name="a"))

    drug_counts = (df
                   .groupby("drugname")["primaryid"]
                   .nunique()
                   .reset_index(name="n_drug"))

    reac_counts = (df
                   .groupby("pt")["primaryid"]
                   .nunique()
                   .reset_index(name="n_reac"))

    ror_df = pair_counts.merge(drug_counts, on="drugname").merge(reac_counts, on="pt")
    ror_df["b"] = ror_df["n_drug"] - ror_df["a"]
    ror_df["c"] = ror_df["n_reac"] - ror_df["a"]
    ror_df["d"] = N - ror_df["n_drug"] - ror_df["n_reac"] + ror_df["a"]

    # ROR = (a/b) / (c/d) = (a*d) / (b*c), with Laplace smoothing
    a, b, c, d = ror_df["a"], ror_df["b"], ror_df["c"], ror_df["d"]
    ror_df["ror"] = (a * d) / ((b + 0.5) * (c + 0.5))
    ror_df["ror_lower_ci"] = np.exp(
        np.log(ror_df["ror"]) - 1.96 * np.sqrt(1/a + 1/b + 1/c + 1/d)
    )
    ror_df["n_reports"] = ror_df["a"]
    ror_df["label"] = (
        (ror_df["ror"] >= 2.0) &
        (ror_df["ror_lower_ci"] > 1.0) &
        (ror_df["n_reports"] >= 3)
    ).astype(int)

    return ror_df[["drugname", "pt", "n_reports", "ror", "ror_lower_ci", "label"]]


def stage_join() -> None:
    for name, path in [
        ("cases_deduped", PROCESSED / "cases_deduped.parquet"),
        ("drug_ps",       PROCESSED / "drug_ps.parquet"),
        ("reac",          PROCESSED / "reac.parquet"),
        ("drug_smiles_map", PROCESSED / "drug_smiles_map.parquet"),
    ]:
        if not path.exists():
            print(f"[join] {path} missing — run prior stages first", file=sys.stderr)
            sys.exit(1)

    print("[join] Loading tables …")
    demo  = pd.read_parquet(PROCESSED / "cases_deduped.parquet")
    drug  = pd.read_parquet(PROCESSED / "drug_ps.parquet")
    reac  = pd.read_parquet(PROCESSED / "reac.parquet")
    smiles = pd.read_parquet(PROCESSED / "drug_smiles_map.parquet")

    # normalise join keys to string
    for df in (demo, drug, reac):
        df["primaryid"] = df["primaryid"].astype(str)
    smiles["drugname"] = smiles["drugname"].str.upper()

    # join DEMO + DRUG(PS) + SMILES
    print("[join] Merging demo × drug_ps …")
    df = demo.merge(drug[["primaryid", "drugname", "role_cod", "quarter", "year"]],
                    on="primaryid", how="inner", suffixes=("", "_drug"))
    df["drugname"] = df["drugname"].str.upper()
    df = df.merge(smiles, on="drugname", how="left")
    df["smiles"] = df["smiles"].fillna("[UNK-MOL]")

    # join REAC — explodes to one row per (case, reaction) pair
    print("[join] Merging × reac …")
    reac = reac[["primaryid", "pt"]].dropna(subset=["pt"])
    reac["primaryid"] = reac["primaryid"].astype(str)
    df = df.merge(reac, on="primaryid", how="inner")

    # NLP text input: fallback = drugname + reaction term
    openfda_path = PROCESSED / "openfda_narr.parquet"
    if openfda_path.exists():
        narr = pd.read_parquet(openfda_path)
        narr["primaryid"] = narr["primaryid"].astype(str)
        df = df.merge(narr[["primaryid", "narr_text"]], on="primaryid", how="left")
        df["text_input"] = df["narr_text"].fillna(df["drugname"] + " caused " + df["pt"])
        df = df.drop(columns=["narr_text"])
        print("[join] openFDA NARR enrichment applied")
    else:
        df["text_input"] = df["drugname"] + " caused " + df["pt"]
        print("[join] Using fallback text_input (drugname + pt)")

    # compute ROR labels
    print("[join] Computing ROR labels …")
    ror_df = compute_ror(df)
    df = df.merge(ror_df[["drugname", "pt", "n_reports", "ror", "ror_lower_ci", "label"]],
                  on=["drugname", "pt"], how="left")
    df["label"] = df["label"].fillna(0).astype(int)

    # resolve year col (prefer demo year)
    if "year_drug" in df.columns:
        df = df.drop(columns=["year_drug"])
    if "year" not in df.columns:
        df["year"] = df["quarter"].str[:4].astype(int)

    out_path = PROCESSED / "master.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[join] master.parquet: {len(df):,} rows, {df['label'].mean():.1%} positive → {out_path}")

    print("[join] Next: python clean_faers.py --stage split")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — split (streamed from master.parquet, one row group at a time)
# ─────────────────────────────────────────────────────────────────────────────
def split_bounds() -> dict[str, tuple[int, int]]:
    """Inclusive year ranges per split, from pharmviginet.config (time split — hard constraint)."""
    return {"train": (0, TRAIN_MAX_YEAR), "val": VAL_YEARS, "test": (TEST_MIN_YEAR, 9999)}


def _year_range(pf: pq.ParquetFile, rg: int) -> Optional[tuple[int, int]]:
    col = pf.schema_arrow.get_field_index("year")
    st = pf.metadata.row_group(rg).column(col).statistics
    return (int(st.min), int(st.max)) if st is not None and st.has_min_max else None


def _schema_desc(schema: pa.Schema) -> list[list[str]]:
    return [[f.name, str(f.type)] for f in schema]


def write_splits(src: Path, paths: dict[str, Path], bounds: dict[str, tuple[int, int]],
                 manifest: Path) -> dict:
    """
    Stream src into one parquet file per split, reading one row group at a time in file
    order. Each split is written to a temp file and checked (row totals, year bounds,
    non-empty, schema). Only then are the files published, and the manifest is written
    last: a split without a consistent manifest is incomplete (see check_split).
    """
    pf = pq.ParquetFile(src)
    schema = pf.schema_arrow
    tmp = {k: p.with_name(f".tmp-{p.name}") for k, p in paths.items()}
    for p in tmp.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    writers = {k: pq.ParquetWriter(tmp[k], schema) for k in paths}
    rows = {k: 0 for k in paths}
    years: dict[str, dict[int, int]] = {k: {} for k in paths}
    try:
        for rg in range(pf.num_row_groups):
            yr = _year_range(pf, rg)
            wanted = [k for k, (lo, hi) in bounds.items()
                      if yr is None or not (yr[1] < lo or yr[0] > hi)]
            if not wanted:
                continue
            t = pf.read_row_group(rg)
            year = t.column("year")
            for k in wanted:
                lo, hi = bounds[k]
                part = t.filter(pc.and_(pc.greater_equal(year, lo), pc.less_equal(year, hi)))
                if part.num_rows:
                    writers[k].write_table(part)
                    rows[k] += part.num_rows
                    for v in pc.value_counts(part.column("year")).to_pylist():
                        years[k][v["values"]] = years[k].get(v["values"], 0) + v["counts"]
            del t, year
        for w in writers.values():
            w.close()

        errors = []
        n_src = pf.metadata.num_rows
        if sum(rows.values()) != n_src:
            errors.append(f"split rows {sum(rows.values()):,} != master rows {n_src:,}")
        for k, (lo, hi) in bounds.items():
            if rows[k] == 0:
                errors.append(f"{k}: no rows")
                continue
            out = pq.ParquetFile(tmp[k])
            if out.metadata.num_rows != rows[k]:
                errors.append(f"{k}: file rows {out.metadata.num_rows:,} != written {rows[k]:,}")
            if not out.schema_arrow.equals(schema):
                errors.append(f"{k}: schema differs from master")
            if min(years[k]) < lo or max(years[k]) > hi:
                errors.append(f"{k}: years {min(years[k])}–{max(years[k])} outside {lo}–{hi}")
        if errors:
            raise ValueError("; ".join(errors))
    except BaseException:
        for w in writers.values():
            try:
                w.close()
            except Exception:
                pass
        for p in tmp.values():
            p.unlink(missing_ok=True)
        raise

    # publish: drop the old marker first, then the files, then the new manifest
    manifest.unlink(missing_ok=True)
    for k in paths:
        os.replace(tmp[k], paths[k])
    info = {
        "source": str(src), "source_rows": pf.metadata.num_rows,
        "schema": _schema_desc(schema),
        "splits": {k: {"file": paths[k].name, "rows": rows[k], "year_bounds": list(bounds[k]),
                       "year_min": min(years[k]), "year_max": max(years[k]),
                       "rows_by_year": {str(y): n for y, n in sorted(years[k].items())}}
                   for k in paths},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp_manifest = manifest.with_name(f".tmp-{manifest.name}")
    tmp_manifest.write_text(json.dumps(info, indent=2))
    os.replace(tmp_manifest, manifest)
    return info


def check_split(paths: dict[str, Path], manifest: Path, src: Optional[Path] = None) -> list[str]:
    """Problems with a published split; empty list = complete and consistent."""
    if not manifest.exists():
        return [f"{manifest.name} missing — split incomplete"]
    info = json.loads(manifest.read_text())
    problems = []
    if src is not None and pq.ParquetFile(src).metadata.num_rows != info["source_rows"]:
        problems.append("master row count changed since the split was written")
    if sum(v["rows"] for v in info["splits"].values()) != info["source_rows"]:
        problems.append("manifest split rows do not add up to source rows")
    for k, p in paths.items():
        m = info["splits"].get(k)
        if m is None or not p.exists():
            problems.append(f"{k}: missing from manifest or disk")
            continue
        f = pq.ParquetFile(p)
        if f.metadata.num_rows != m["rows"]:
            problems.append(f"{k}: file rows {f.metadata.num_rows:,} != manifest {m['rows']:,}")
        if _schema_desc(f.schema_arrow) != info["schema"]:
            problems.append(f"{k}: schema differs from manifest")
        lo, hi = m["year_bounds"]
        if m["year_min"] < lo or m["year_max"] > hi:
            problems.append(f"{k}: years outside bounds")
    return problems


def stage_split() -> None:
    if not MASTER_PARQUET.exists():
        print(f"[split] {MASTER_PARQUET} missing — run --stage join first", file=sys.stderr)
        sys.exit(1)
    paths = {"train": TRAIN_PARQUET, "val": VAL_PARQUET, "test": TEST_PARQUET}
    bounds = split_bounds()
    print(f"[split] {MASTER_PARQUET.name} → " + ", ".join(f"{k} {lo}–{hi}" for k, (lo, hi) in bounds.items()))
    try:
        info = write_splits(MASTER_PARQUET, paths, bounds, SPLIT_MANIFEST)
    except ValueError as e:
        print(f"[split] ERROR: {e} — nothing published", file=sys.stderr)
        sys.exit(1)
    problems = check_split(paths, SPLIT_MANIFEST, MASTER_PARQUET)
    if problems:
        print("[split] ERROR: " + "; ".join(problems), file=sys.stderr)
        sys.exit(1)
    for k, v in info["splits"].items():
        print(f"[split] {k}: {v['rows']:,} rows, years {v['year_min']}–{v['year_max']} → {paths[k]}")
    print(f"[split] manifest → {SPLIT_MANIFEST}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="PharmVigiNet pipeline stages 2–6")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage", choices=["clean", "dedup", "smiles", "join", "split"],
                       help="Run a single stage")
    group.add_argument("--all", action="store_true",
                       help="Run all stages 2→3→4→5→6 in sequence")
    parser.add_argument("--table", nargs="+",
                        help="(--stage clean only) restrict to specific tables")
    args = parser.parse_args()

    if args.all:
        print("=== Stage 2: clean ===")
        stage_clean()
        print("\n=== Stage 3: dedup ===")
        stage_dedup()
        print("\n=== Stage 4: smiles ===")
        stage_smiles()
        print("\n=== Stage 5: join ===")
        stage_join()
        print("\n=== Stage 6: split ===")
        stage_split()
    elif args.stage == "clean":
        tables = [t.upper() for t in args.table] if args.table else None
        stage_clean(tables)
    elif args.stage == "dedup":
        stage_dedup()
    elif args.stage == "smiles":
        stage_smiles()
    elif args.stage == "join":
        stage_join()
    elif args.stage == "split":
        stage_split()


if __name__ == "__main__":
    main()
