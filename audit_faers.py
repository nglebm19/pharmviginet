#!/usr/bin/env python3
"""
audit_faers.py — Complete FAERS/AERS data audit and validation

Usage:
    python audit_faers.py                    # full audit
    python audit_faers.py --quick            # file presence only, no row counts
    python audit_faers.py --quarter 2024Q4   # audit single quarter in detail
    python audit_faers.py --show-schema      # schema diff table only
    python audit_faers.py --narr-only        # NARR coverage analysis only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
EXTRACT_DIR    = Path("data/extracted")
LOG_DIR        = Path("data/logs")
AUDIT_REPORT   = LOG_DIR / "audit_report.json"
AUDIT_SUMMARY  = LOG_DIR / "audit_summary.txt"
SCHEMA_MAP_OUT = LOG_DIR / "schema_map.json"

# ── table definitions ──────────────────────────────────────────────────────────
CORE_TABLES     = ["DEMO", "DRUG", "REAC"]      # missing any → CRITICAL
STANDARD_TABLES = ["DEMO", "DRUG", "REAC", "INDI", "OUTC", "RPSR"]
ALL_TABLES      = STANDARD_TABLES + ["THER", "NARR"]

# Public FAERS downloads include THER (therapy dates) but NOT NARR (narratives).
# NARR is only available via FDA direct submission requests.
NARR_NLP_THRESHOLD = 0.10  # 10% coverage = usable for NLP

# Known column renames AERS (uppercase) → FAERS (lowercase)
KNOWN_RENAMES: dict[str, dict[str, str]] = {
    "DEMO": {"isr": "primaryid", "case": "caseid", "gndr_cod": "sex"},
    "DRUG": {"isr": "primaryid", "drug_seq": "drug_seq"},   # drug_seq unchanged
    "REAC": {"isr": "primaryid"},
    "INDI": {"isr": "primaryid", "drug_seq": "indi_drug_seq"},
    "OUTC": {"isr": "primaryid"},
    "RPSR": {"isr": "primaryid"},
    "THER": {"isr": "primaryid", "drug_seq": "dsg_drug_seq"},
    "NARR": {"isr": "primaryid"},
}

console = Console(highlight=False)


# ── era / quarter helpers ──────────────────────────────────────────────────────

def is_aers(year: int, q: int) -> bool:
    """AERS era: 2004Q1 through 2012Q3.  2012Q4 onward is FAERS."""
    return year < 2012 or (year == 2012 and q <= 3)


def parse_quarter_name(name: str) -> tuple[int, int] | None:
    m = re.fullmatch(r"(\d{4})Q([1-4])", name, re.IGNORECASE)
    return (int(m.group(1)), int(m.group(2))) if m else None


# ── file discovery ─────────────────────────────────────────────────────────────

def _ascii_dir(qdir: Path) -> Path:
    """Return the ascii/ subdirectory (case-insensitive), falling back to qdir."""
    for child in qdir.iterdir():
        if child.is_dir() and child.name.lower() == "ascii":
            return child
    return qdir


def find_table_file(qdir: Path, table: str, year: int, q: int) -> Path | None:
    """
    Find {TABLE}{YY}Q{N}[optional_suffix].{txt|TXT} regardless of case.
    Handles non-standard filenames like DEMO18Q1_new.txt via prefix match.
    AERS uses uppercase .TXT, FAERS uses lowercase .txt.
    """
    adir = _ascii_dir(qdir)
    yy   = str(year)[2:]
    prefix = f"{table}{yy}Q{q}".upper()
    for p in adir.iterdir():
        if p.is_file() and p.stem.upper().startswith(prefix) and p.suffix.lower() == ".txt":
            return p
    return None


def discover_quarters() -> list[dict[str, Any]]:
    if not EXTRACT_DIR.is_dir():
        console.print(f"[red]ERROR: {EXTRACT_DIR} does not exist. Run from repo root.[/red]")
        sys.exit(1)
    quarters: list[dict[str, Any]] = []
    for d in sorted(EXTRACT_DIR.iterdir()):
        if not d.is_dir():
            continue
        parsed = parse_quarter_name(d.name)
        if not parsed:
            continue
        year, q = parsed
        quarters.append({
            "name":    d.name,
            "year":    year,
            "quarter": q,
            "era":     "aers" if is_aers(year, q) else "faers",
            "path":    d,
        })
    return quarters


# ── fast row counting ──────────────────────────────────────────────────────────

def count_lines_fast(path: Path) -> int:
    """Count newline bytes via buffered binary read — handles multi-GB files quickly."""
    n = 0
    with path.open("rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            n += chunk.count(b"\n")
    return n


def data_rows(path: Path) -> int:
    """Line count minus 1 header row; minimum 0."""
    return max(0, count_lines_fast(path) - 1)


# ── schema / readability check ─────────────────────────────────────────────────

def check_schema(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False, "columns": [], "phantom_col": False, "error": None
    }
    try:
        df = pd.read_csv(
            path, sep="$", encoding="latin1",
            nrows=10, low_memory=False,
            on_bad_lines="skip",
        )
        cols = [c.strip() for c in df.columns]
        phantom = bool(
            cols and (cols[-1] == "" or str(cols[-1]).startswith("Unnamed:"))
        )
        result.update({"ok": True, "columns": cols, "phantom_col": phantom})
    except Exception as exc:
        result["error"] = str(exc)[:200]
    return result


# ── presence check ─────────────────────────────────────────────────────────────

def check_presence(qinfo: dict[str, Any]) -> dict[str, Any]:
    year, q = qinfo["year"], qinfo["quarter"]
    presence:   dict[str, bool]      = {}
    file_paths: dict[str, Path | None] = {}

    for t in ALL_TABLES:
        p  = find_table_file(qinfo["path"], t, year, q)
        ok = p is not None and p.stat().st_size > 0
        presence[t]   = ok
        file_paths[t] = p if ok else None

    missing_core = [t for t in CORE_TABLES if not presence[t]]
    issues:   list[str] = []
    warnings: list[str] = []

    if missing_core:
        issues.append(f"Missing CORE tables: {', '.join(missing_core)}")
    for t in ["INDI", "OUTC", "RPSR"]:
        if not presence[t]:
            warnings.append(f"Missing {t}")
    if not presence["NARR"]:
        warnings.append("NARR absent — not included in public FAERS downloads")
    if not presence["THER"]:
        warnings.append("Missing THER")

    return {
        "presence":   presence,
        "file_paths": file_paths,
        "issues":     issues,
        "warnings":   warnings,
    }


# ── row count flags ────────────────────────────────────────────────────────────

def row_count_flags(
    qinfo:     dict[str, Any],
    presence:  dict[str, bool],
    fpaths:    dict[str, Path | None],
    counts:    dict[str, int],
) -> tuple[list[str], list[str]]:
    """Returns (issues, warnings) based on row count anomalies."""
    issues:   list[str] = []
    warnings: list[str] = []
    demo = counts.get("DEMO", 0)
    drug = counts.get("DRUG", 0)
    reac = counts.get("REAC", 0)
    narr = counts.get("NARR", 0)

    if presence["DEMO"] and demo < 1_000:
        issues.append(f"DEMO rows suspiciously low: {demo:,} (expected ≥ 1,000)")
    if presence["DEMO"] and presence["DRUG"] and drug < demo:
        warnings.append(f"DRUG rows ({drug:,}) < DEMO rows ({demo:,})")
    if presence["DEMO"] and presence["REAC"] and reac < demo:
        warnings.append(f"REAC rows ({reac:,}) < DEMO rows ({demo:,})")
    if presence["NARR"] and demo > 0 and narr > demo:
        issues.append(f"NARR rows ({narr:,}) > DEMO rows ({demo:,}) — impossible")
    if presence["NARR"] and demo > 0 and qinfo["year"] > 2015:
        cov = narr / demo
        if cov < 0.05:
            warnings.append(f"NARR coverage very low post-2015: {cov:.1%}")

    return issues, warnings


# ── schema comparison ──────────────────────────────────────────────────────────

def build_schema_diffs(
    quarters:     list[dict[str, Any]],
    schema_cache: dict[str, dict[str, list[str]]],
) -> list[dict[str, Any]]:
    """Collect column-name differences between AERS and FAERS eras."""
    per_table: dict[str, dict[str, set[str]]] = {
        t: {"aers": set(), "faers": set()} for t in ALL_TABLES
    }
    for q in quarters:
        era = q["era"]
        for t in ALL_TABLES:
            cols = schema_cache.get(q["name"], {}).get(t)
            if cols:
                per_table[t][era].update(c.lower() for c in cols)

    diffs: list[dict[str, Any]] = []
    for t in ALL_TABLES:
        aers_cols  = per_table[t]["aers"]
        faers_cols = per_table[t]["faers"]
        if not aers_cols or not faers_cols:
            continue
        known      = KNOWN_RENAMES.get(t, {})
        aers_only  = sorted(aers_cols - faers_cols)
        faers_only = sorted(faers_cols - aers_cols)
        explained  = set(known.values())

        for ac in aers_only:
            diffs.append({
                "table":     t,
                "aers_col":  ac,
                "faers_col": known.get(ac, "—"),
                "change":    "renamed" if ac in known else "removed_in_faers",
            })
        for fc in faers_only:
            if fc not in explained:
                diffs.append({
                    "table":     t,
                    "aers_col":  "—",
                    "faers_col": fc,
                    "change":    "added_in_faers",
                })

    return sorted(diffs, key=lambda x: (x["table"], x["change"], x["aers_col"]))


# ── NARR coverage ──────────────────────────────────────────────────────────────

def narr_coverage(
    quarters:   list[dict[str, Any]],
    counts_map: dict[str, dict[str, int]],
) -> dict[str, Any]:
    by_qtr:   dict[str, float]      = {}
    by_year:  dict[int, list[float]] = defaultdict(list)
    crossings = {"10pct": None, "25pct": None, "50pct": None}

    for q in quarters:
        qn   = q["name"]
        demo = counts_map.get(qn, {}).get("DEMO", 0)
        narr = counts_map.get(qn, {}).get("NARR", 0)
        cov  = narr / demo if demo > 0 else 0.0
        by_qtr[qn] = round(cov, 4)
        by_year[q["year"]].append(cov)
        for thresh, key in [(0.10, "10pct"), (0.25, "25pct"), (0.50, "50pct")]:
            if crossings[key] is None and cov >= thresh:
                crossings[key] = qn

    return {
        "by_quarter": by_qtr,
        "avg_by_year": {
            yr: round(sum(v) / len(v), 4)
            for yr, v in sorted(by_year.items())
        },
        "crossings": crossings,
    }


# ── safe-for-use classification ────────────────────────────────────────────────

def classify(
    quarters:      list[dict[str, Any]],
    presence_map:  dict[str, dict[str, bool]],
    narr_by_qtr:   dict[str, float],
) -> tuple[list[str], list[str]]:
    safe_nlp: list[str] = []
    safe_ml:  list[str] = []
    for q in quarters:
        qn   = q["name"]
        pres = presence_map.get(qn, {})
        if all(pres.get(t) for t in CORE_TABLES):
            safe_ml.append(qn)
            if narr_by_qtr.get(qn, 0.0) >= NARR_NLP_THRESHOLD:
                safe_nlp.append(qn)
    return safe_nlp, safe_ml


# ── terminal output helpers ────────────────────────────────────────────────────

def print_schema_table(diffs: list[dict[str, Any]]) -> None:
    tbl = Table(title="Schema Differences: AERS → FAERS", header_style="bold magenta")
    tbl.add_column("Table",       style="bold")
    tbl.add_column("AERS column", style="yellow")
    tbl.add_column("FAERS column", style="cyan")
    tbl.add_column("Change")
    COLORS = {
        "renamed":         "yellow",
        "removed_in_faers": "red",
        "added_in_faers":   "green",
    }
    for d in diffs:
        color = COLORS.get(d["change"], "white")
        tbl.add_row(
            d["table"], d["aers_col"], d["faers_col"],
            f"[{color}]{d['change']}[/{color}]",
        )
    console.print(tbl)


def print_narr_table(cov: dict[str, Any]) -> None:
    tbl = Table(title="NARR Coverage by Year", header_style="bold cyan")
    tbl.add_column("Year")
    tbl.add_column("Avg NARR%")
    tbl.add_column("NLP-usable (≥10%)?")
    for yr, avg in cov["avg_by_year"].items():
        pct    = avg * 100
        usable = "[green]YES[/green]" if avg >= NARR_NLP_THRESHOLD else "[red]NO[/red]"
        tbl.add_row(str(yr), f"{pct:.1f}%", usable)
    console.print(tbl)
    console.print()
    console.print("[bold]NARR coverage milestones:[/bold]")
    for key, qn in cov["crossings"].items():
        label = key.replace("pct", "%")
        val   = f"[green]{qn}[/green]" if qn else "[red]never reached[/red]"
        console.print(f"  ≥{label} first reached: {val}")


def print_dashboard(
    quarters:     list[dict[str, Any]],
    presence_map: dict[str, dict[str, bool]],
    issues_map:   dict[str, list[str]],
    warnings_map: dict[str, list[str]],
    counts_map:   dict[str, dict[str, int]],
    narr_by_qtr:  dict[str, float],
    full: bool = True,
) -> None:
    tbl = Table(
        title="FAERS/AERS Audit Dashboard",
        header_style="bold cyan",
        show_lines=False,
    )
    for col in ["Quarter", "Era", "DEMO", "DRUG", "REAC", "INDI", "OUTC", "RPSR", "THER", "NARR", "NARR%", "Status"]:
        tbl.add_column(col, no_wrap=True)

    for q in quarters:
        qn   = q["name"]
        pres = presence_map.get(qn, {})
        iss  = issues_map.get(qn, [])
        warn = warnings_map.get(qn, [])
        cnts = counts_map.get(qn, {}) if full else {}

        def cell(t: str) -> str:
            if not pres.get(t):
                return "[dim]—[/dim]"
            if not full:
                return "[green]✓[/green]"
            n = cnts.get(t, 0)
            return f"{n:,}" if n else "[yellow]0[/yellow]"

        cov_s = (
            f"{narr_by_qtr.get(qn, 0) * 100:.1f}%"
            if full and pres.get("NARR") else "—"
        )
        if iss:
            rc, st = "red",    "[red]CRITICAL[/red]"
        elif warn:
            rc, st = "yellow", "[yellow]WARN[/yellow]"
        else:
            rc, st = "green",  "[green]OK[/green]"

        tbl.add_row(
            f"[{rc}]{qn}[/{rc}]",
            q["era"].upper(),
            cell("DEMO"), cell("DRUG"), cell("REAC"),
            cell("INDI"), cell("OUTC"), cell("RPSR"),
            cell("THER"), cell("NARR"),
            cov_s, st,
        )
    console.print(tbl)


def print_summary_box(
    quarters:     list[dict[str, Any]],
    issues_map:   dict[str, list[str]],
    warnings_map: dict[str, list[str]],
    counts_map:   dict[str, dict[str, int]],
    safe_nlp:     list[str],
    safe_ml:      list[str],
) -> None:
    total       = len(quarters)
    n_crit      = sum(1 for q in quarters if issues_map.get(q["name"]))
    n_warn      = sum(1 for q in quarters if warnings_map.get(q["name"]) and not issues_map.get(q["name"]))
    n_ok        = total - n_crit - n_warn
    total_cases = sum(c.get("DEMO", 0) for c in counts_map.values())
    era_aers    = sum(1 for q in quarters if q["era"] == "aers")
    era_faers   = sum(1 for q in quarters if q["era"] == "faers")

    W = 55
    def row(s: str) -> str:
        return ("║  " + s).ljust(W - 1) + "║"

    lines = [
        "╔" + "═" * (W - 2) + "╗",
        row("FAERS/AERS Audit Complete"),
        "╠" + "═" * (W - 2) + "╣",
        row(f"Total quarters audited   : {total}"),
        row(f"Era breakdown            : AERS={era_aers}  FAERS={era_faers}"),
        row(f"Fully healthy (OK)       : {n_ok}"),
        row(f"Warnings only            : {n_warn}"),
        row(f"Critical issues          : {n_crit}"),
        row(f"Total cases (DEMO rows)  : ~{total_cases:,}"),
        row(f"Safe for ML training     : {len(safe_ml)} quarters"),
        row(f"Safe for NLP (NARR≥10%) : {len(safe_nlp)} quarters"),
        "╚" + "═" * (W - 2) + "╝",
    ]
    console.print("\n".join(lines))


# ── single-quarter detail mode ─────────────────────────────────────────────────

def audit_single_quarter(qname: str) -> None:
    quarters = [q for q in discover_quarters() if q["name"].upper() == qname.upper()]
    if not quarters:
        console.print(f"[red]Quarter {qname!r} not found in {EXTRACT_DIR}[/red]")
        sys.exit(1)
    q = quarters[0]
    console.rule(f"[bold]Detail: {q['name']}  era={q['era'].upper()}[/bold]")

    pres_result = check_presence(q)
    pres        = pres_result["presence"]
    fpaths      = pres_result["file_paths"]

    tbl = Table(title=f"Files in {q['name']}", header_style="bold")
    tbl.add_column("Table", style="bold")
    tbl.add_column("Present")
    tbl.add_column("Size (MB)", justify="right")
    tbl.add_column("Data rows", justify="right")
    tbl.add_column("PK column")
    tbl.add_column("Phantom col")
    tbl.add_column("Columns (first 8)", no_wrap=False, max_width=60)

    for t in ALL_TABLES:
        p = fpaths.get(t)
        if not pres.get(t) or not p:
            tbl.add_row(t, "[red]✗[/red]", "—", "—", "—", "—", "—")
            continue
        size_mb = f"{p.stat().st_size / 1e6:.1f}"
        rows    = f"{data_rows(p):,}"
        s       = check_schema(p)
        if not s["ok"]:
            tbl.add_row(t, "[red]read error[/red]", size_mb, rows, "—", "—", s["error"] or "")
            continue
        cols_lo  = [c.lower() for c in s["columns"]]
        pk       = (
            "primaryid" if "primaryid" in cols_lo else
            "isr"       if "isr"       in cols_lo else
            "[red]MISSING[/red]"
        )
        phantom  = "[red]YES[/red]" if s["phantom_col"] else "no"
        col_str  = ", ".join(s["columns"][:8])
        if len(s["columns"]) > 8:
            col_str += f"  …(+{len(s['columns'])-8})"
        tbl.add_row(t, "[green]✓[/green]", size_mb, rows, pk, phantom, col_str)

    console.print(tbl)
    for iss in pres_result["issues"]:
        console.print(f"[red]  CRITICAL: {iss}[/red]")
    for w in pres_result["warnings"]:
        console.print(f"[yellow]  WARNING : {w}[/yellow]")


# ── save reports ───────────────────────────────────────────────────────────────

def save_reports(
    quarters:     list[dict[str, Any]],
    presence_map: dict[str, dict[str, bool]],
    counts_map:   dict[str, dict[str, int]],
    narr_info:    dict[str, Any],
    schema_diffs: list[dict[str, Any]],
    issues_map:   dict[str, list[str]],
    warnings_map: dict[str, list[str]],
    safe_nlp:     list[str],
    safe_ml:      list[str],
    schema_cache: dict[str, dict[str, list[str]]],
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    era_aers  = sum(1 for q in quarters if q["era"] == "aers")
    era_faers = sum(1 for q in quarters if q["era"] == "faers")

    report: dict[str, Any] = {
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_quarters":  len(quarters),
        "era_breakdown":   {"aers": era_aers, "faers": era_faers},
        "file_presence": {
            q["name"]: {t: bool(v) for t, v in presence_map[q["name"]].items()}
            for q in quarters
        },
        "row_counts": {
            q["name"]: counts_map.get(q["name"], {}) for q in quarters
        },
        "narr_coverage":  narr_info["by_quarter"],
        "narr_by_year":   narr_info["avg_by_year"],
        "narr_crossings": narr_info["crossings"],
        "schema_differences": schema_diffs,
        "critical_issues": {
            q["name"]: issues_map[q["name"]]
            for q in quarters if issues_map.get(q["name"])
        },
        "warnings": {
            q["name"]: warnings_map[q["name"]]
            for q in quarters if warnings_map.get(q["name"])
        },
        "quarters_safe_for_nlp": safe_nlp,
        "quarters_safe_for_ml":  safe_ml,
    }
    AUDIT_REPORT.write_text(json.dumps(report, indent=2))

    SCHEMA_MAP_OUT.write_text(json.dumps(schema_cache, indent=2))

    summary_lines = [
        "FAERS/AERS Audit Summary",
        f"Generated: {report['audit_timestamp']}",
        "=" * 60,
        f"Total quarters : {len(quarters)} (AERS={era_aers}, FAERS={era_faers})",
        f"Safe for ML    : {len(safe_ml)} quarters",
        f"Safe for NLP   : {len(safe_nlp)} quarters  (NARR ≥ 10%)",
        "",
        "NARR coverage milestones:",
    ]
    for key, qn in narr_info["crossings"].items():
        summary_lines.append(f"  ≥{key.replace('pct','%')}: {qn or 'never reached'}")
    summary_lines += ["", "Schema differences (AERS → FAERS):"]
    for d in schema_diffs:
        summary_lines.append(
            f"  [{d['table']}] {d['aers_col']} → {d['faers_col']}  ({d['change']})"
        )
    summary_lines += ["", "Critical issues:"]
    for qn, iss in report["critical_issues"].items():
        for i in iss:
            summary_lines.append(f"  {qn}: {i}")
    AUDIT_SUMMARY.write_text("\n".join(summary_lines))

    console.print(f"[dim]Saved: {AUDIT_REPORT}[/dim]")
    console.print(f"[dim]Saved: {SCHEMA_MAP_OUT}[/dim]")
    console.print(f"[dim]Saved: {AUDIT_SUMMARY}[/dim]")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="FAERS/AERS data audit and validation")
    ap.add_argument("--quick",       action="store_true", help="File presence only, skip row counts")
    ap.add_argument("--quarter",     metavar="YYYYQN",    help="Audit one quarter in detail")
    ap.add_argument("--show-schema", action="store_true", help="Print AERS→FAERS schema diff only")
    ap.add_argument("--narr-only",   action="store_true", help="NARR coverage analysis only")
    args = ap.parse_args()

    if args.quarter:
        audit_single_quarter(args.quarter)
        return

    # ── Step 1: discover ──────────────────────────────────────────────────────
    quarters = discover_quarters()
    if not quarters:
        console.print(f"[red]No YYYYQN folders found in {EXTRACT_DIR}[/red]")
        sys.exit(1)
    era_aers  = sum(1 for q in quarters if q["era"] == "aers")
    era_faers = sum(1 for q in quarters if q["era"] == "faers")
    console.print(
        f"[bold]Discovered {len(quarters)} quarters[/bold]  "
        f"({quarters[0]['name']} → {quarters[-1]['name']})  "
        f"[dim]AERS={era_aers} FAERS={era_faers}[/dim]"
    )

    # ── Step 2: file presence ─────────────────────────────────────────────────
    presence_map:  dict[str, dict[str, bool]]       = {}
    file_paths_map: dict[str, dict[str, Path | None]] = {}
    issues_map:    dict[str, list[str]]             = {}
    warnings_map:  dict[str, list[str]]             = {}

    console.print("[bold]Step 1 — File presence check...[/bold]")
    for q in tqdm(quarters, desc="Presence", ncols=90):
        r = check_presence(q)
        presence_map[q["name"]]    = r["presence"]
        file_paths_map[q["name"]]  = r["file_paths"]
        issues_map[q["name"]]      = r["issues"]
        warnings_map[q["name"]]    = r["warnings"]

    # ── Step 3: schema / readability ──────────────────────────────────────────
    schema_cache: dict[str, dict[str, list[str]]] = {}
    console.print("[bold]Step 2 — Schema check (first 10 rows each)...[/bold]")
    for q in tqdm(quarters, desc="Schema", ncols=90):
        qn   = q["name"]
        qtbl = {}
        for t in ALL_TABLES:
            p = file_paths_map[qn].get(t)
            if not p:
                continue
            s = check_schema(p)
            if s["ok"]:
                qtbl[t] = s["columns"]
                if s["phantom_col"]:
                    warnings_map[qn].append(f"{t} has phantom trailing column (trailing $)")
            else:
                issues_map[qn].append(f"{t} unreadable: {s['error']}")
        schema_cache[qn] = qtbl

    schema_diffs = build_schema_diffs(quarters, schema_cache)

    if args.show_schema:
        print_schema_table(schema_diffs)
        return

    # ── Step 4: row counts ────────────────────────────────────────────────────
    counts_map: dict[str, dict[str, int]] = {}

    if not args.quick:
        console.print("[bold]Step 3 — Row counts (binary line scan)...[/bold]")
        for q in tqdm(quarters, desc="Counting", ncols=90):
            qn  = q["name"]
            cnts: dict[str, int] = {}
            for t in ALL_TABLES:
                p = file_paths_map[qn].get(t)
                cnts[t] = data_rows(p) if p else 0
            counts_map[qn] = cnts
            iss, warn = row_count_flags(q, presence_map[qn], file_paths_map[qn], cnts)
            issues_map[qn].extend(iss)
            warnings_map[qn].extend(warn)
    else:
        for q in quarters:
            counts_map[q["name"]] = {}

    # ── Step 5: NARR coverage ─────────────────────────────────────────────────
    narr_info = narr_coverage(quarters, counts_map)

    if args.narr_only:
        print_narr_table(narr_info)
        return

    # ── classify ──────────────────────────────────────────────────────────────
    safe_nlp, safe_ml = classify(quarters, presence_map, narr_info["by_quarter"])

    # ── Step 6: save reports ──────────────────────────────────────────────────
    if not args.quick:
        console.print("[bold]Step 4 — Saving reports...[/bold]")
        save_reports(
            quarters, presence_map, counts_map,
            narr_info, schema_diffs,
            issues_map, warnings_map,
            safe_nlp, safe_ml, schema_cache,
        )

    # ── print results ─────────────────────────────────────────────────────────
    console.print()
    print_dashboard(
        quarters, presence_map, issues_map, warnings_map,
        counts_map, narr_info["by_quarter"],
        full=not args.quick,
    )

    if not args.quick:
        console.print()
        print_narr_table(narr_info)
        console.print()
        print_schema_table(schema_diffs)

    console.print()
    print_summary_box(
        quarters, issues_map, warnings_map,
        counts_map, safe_nlp, safe_ml,
    )

    # ── NARR warning ──────────────────────────────────────────────────────────
    if not safe_nlp:
        console.print()
        console.print(
            "[bold red]⚠  NARR WARNING[/bold red]\n"
            "No quarters have NARR (narrative text) files.\n"
            "Public FAERS quarterly downloads do NOT include NARR.\n"
            "To obtain narrative text you must:\n"
            "  1. Submit a FOIA request to FDA for the full FAERS dataset, OR\n"
            "  2. Use the FAERS Public Dashboard API (limited fields), OR\n"
            "  3. Use the openFDA API (includes narratives for some cases).\n"
            "All ML training (non-NLP) is unaffected — DEMO/DRUG/REAC/INDI are present."
        )


if __name__ == "__main__":
    main()
