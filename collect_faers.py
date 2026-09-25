#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time
from datetime import datetime
from typing import Any
from zipfile import BadZipFile, ZipFile

import pandas as pd
import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

BASE_URL = "https://fis.fda.gov/content/Exports"
SOURCE_PAGE = "https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html"
RAW_DIR = Path("data/raw")
EXTRACT_DIR = Path("data/extracted")
LOG_DIR = Path("data/logs")
REQUEST_TIMEOUT = 60
RETRY_COUNT = 3
RETRY_BACKOFF = [2, 4, 8]
POLITE_DELAY = 1.5
LOCK_FILE = LOG_DIR / ".faers_collector.lock"
DOWNLOAD_LOG = LOG_DIR / "download_log.json"
VALIDATION_REPORT = LOG_DIR / "validation_report.json"

EXPECTED_TABLES = ["DEMO", "DRUG", "REAC", "NARR", "OUTC", "RPSR", "INDI"]
FALLBACK_FILENAMES = [
    "aers_ascii_2004q1.zip", "aers_ascii_2004q2.zip", "aers_ascii_2004q3.zip", "aers_ascii_2004q4.zip",
    "aers_ascii_2005q1.zip", "aers_ascii_2005q2.zip", "aers_ascii_2005q3.zip", "aers_ascii_2005q4.zip",
    "aers_ascii_2006q1.zip", "aers_ascii_2006q2.zip", "aers_ascii_2006q3.zip", "aers_ascii_2006q4.zip",
    "aers_ascii_2007q1.zip", "aers_ascii_2007q2.zip", "aers_ascii_2007q3.zip", "aers_ascii_2007q4.zip",
    "aers_ascii_2008q1.zip", "aers_ascii_2008q2.zip", "aers_ascii_2008q3.zip", "aers_ascii_2008q4.zip",
    "aers_ascii_2009q1.zip", "aers_ascii_2009q2.zip", "aers_ascii_2009q3.zip", "aers_ascii_2009q4.zip",
    "aers_ascii_2010q1.zip", "aers_ascii_2010q2.zip", "aers_ascii_2010q3.zip", "aers_ascii_2010q4.zip",
    "aers_ascii_2011q1.zip", "aers_ascii_2011q2.zip", "aers_ascii_2011q3.zip", "aers_ascii_2011q4.zip",
    "aers_ascii_2012q1.zip", "aers_ascii_2012q2.zip", "aers_ascii_2012q3.zip", "faers_ascii_2012q4.zip",
    "faers_ascii_2013q1.zip", "faers_ascii_2013q2.zip", "faers_ascii_2013q3.zip", "faers_ascii_2013q4.zip",
    "faers_ascii_2014q1.zip", "faers_ascii_2014q2.zip", "faers_ascii_2014q3.zip", "faers_ascii_2014q4.zip",
    "faers_ascii_2015q1.zip", "faers_ascii_2015q2.zip", "faers_ascii_2015q3.zip", "faers_ascii_2015q4.zip",
    "faers_ascii_2016q1.zip", "faers_ascii_2016q2.zip", "faers_ascii_2016q3.zip", "faers_ascii_2016q4.zip",
    "faers_ascii_2017q1.zip", "faers_ascii_2017q2.zip", "faers_ascii_2017q3.zip", "faers_ascii_2017q4.zip",
    "faers_ascii_2018q1.zip", "faers_ascii_2018q2.zip", "faers_ascii_2018q3.zip", "faers_ascii_2018q4.zip",
    "faers_ascii_2019Q1.zip", "faers_ascii_2019Q2.zip", "faers_ascii_2019Q3.zip", "faers_ascii_2019Q4.zip",
    "faers_ascii_2020Q1.zip", "faers_ascii_2020Q2.zip", "faers_ascii_2020Q3.zip", "faers_ascii_2020Q4.zip",
    "faers_ascii_2021Q1.zip", "faers_ascii_2021Q2.zip", "faers_ascii_2021Q3.zip", "faers_ascii_2021Q4.zip",
    "faers_ascii_2022q1.zip", "faers_ascii_2022q2.zip", "faers_ascii_2022Q3.zip", "faers_ascii_2022Q4.zip",
    "faers_ascii_2023q1.zip", "faers_ascii_2023q2.zip", "faers_ascii_2023Q3.zip", "faers_ascii_2023Q4.zip",
    "faers_ascii_2024q1.zip", "faers_ascii_2024q2.zip", "faers_ascii_2024q3.zip", "faers_ascii_2024Q4.zip",
    "faers_ascii_2025q1.zip", "faers_ascii_2025q2.zip", "faers_ascii_2025q3.zip", "faers_ascii_2025Q4.zip",
    "faers_ascii_2026q1.zip",
]

console = Console()
logger = logging.getLogger("faers_collector")
_STOP = False


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return
    fh = RotatingFileHandler(LOG_DIR / "faers_collector.log", maxBytes=10 * 1024 * 1024, backupCount=3)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)


def acquire_lock() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        raise SystemExit("Another instance is running (lock file exists).")


def release_lock() -> None:
    if LOCK_FILE.exists():
        LOCK_FILE.unlink(missing_ok=True)


def _signal_handler(signum: int, frame: Any) -> None:
    del signum, frame
    global _STOP
    _STOP = True
    console.print("\n[yellow]Interrupted. Finishing current step and saving progress...[/yellow]")


def _quarter_key(year: int, quarter: int) -> tuple[int, int]:
    return (year, quarter)


def _parse_file_meta(filename: str) -> dict[str, Any] | None:
    m = re.match(r"(aers|faers)_ascii_(\d{4})([Qq])([1-4])\.zip$", filename)
    if not m:
        return None
    era_prefix, year, q_case, q = m.groups()
    return {
        "filename": filename,
        "year": int(year),
        "quarter": int(q),
        "quarter_label": f"{year}Q{q}",
        "era": era_prefix,
        "q_case": q_case,
        "url": f"{BASE_URL}/{filename}",
    }


def discover_urls() -> list[dict[str, Any]]:
    urls: list[dict[str, Any]] = []
    try:
        resp = requests.get(SOURCE_PAGE, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        posted_date: str | None = None
        size_mb: float | None = None
        for node in soup.descendants:
            if isinstance(node, str):
                text = node.strip()
                mp = re.search(r"posted on\s+([0-9]{1,2}-[A-Za-z]{3}-[0-9]{4})", text)
                if mp:
                    posted_date = mp.group(1)
            elif getattr(node, "name", None) == "a":
                txt = node.get_text(" ", strip=True)
                ms = re.search(r"ZIP\s*-\s*([0-9]+(?:\.[0-9]+)?)MB", txt, re.IGNORECASE)
                if ms:
                    size_mb = float(ms.group(1))
                href = node.get("href")
                if not href:
                    continue
                fname = Path(href).name
                meta = _parse_file_meta(fname)
                if meta:
                    meta["size_mb"] = size_mb
                    meta["posted_date"] = posted_date
                    urls.append(meta)
        if not urls:
            raise ValueError("No ASCII URLs found in page")
    except Exception as e:
        logger.exception("Discovery failed, using fallback: %s", e)
        for fname in FALLBACK_FILENAMES:
            meta = _parse_file_meta(fname)
            if meta:
                meta["size_mb"] = None
                meta["posted_date"] = None
                urls.append(meta)

    dedup: dict[str, dict[str, Any]] = {u["filename"]: u for u in urls}
    out = sorted(dedup.values(), key=lambda x: _quarter_key(x["year"], x["quarter"]))
    console.print(f"Found {len(out)} quarterly files spanning {out[0]['year']} to {out[-1]['year']}")
    return out


def _filter_range(urls: list[dict[str, Any]], start_year: int | None, end_year: int | None) -> list[dict[str, Any]]:
    result = []
    for u in urls:
        if start_year is not None and u["year"] < start_year:
            continue
        if end_year is not None and u["year"] > end_year:
            continue
        result.append(u)
    return result


def _get_remote_size(url: str) -> int | None:
    try:
        r = requests.head(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        cl = r.headers.get("Content-Length")
        return int(cl) if cl else None
    except Exception:
        return None


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def download_all(urls: list[dict[str, Any]], dry_run: bool = False, resume: bool = False) -> list[dict[str, Any]]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logs = _read_json_list(DOWNLOAD_LOG)
    prior_ok = {x["quarter"] for x in logs if x.get("status") == "ok"}
    session = requests.Session()
    results = []

    for item in urls:
        if _STOP:
            break
        quarter = item["quarter_label"]
        filename = item["filename"]
        url = item["url"]
        target = RAW_DIR / filename
        remote_size = _get_remote_size(url)

        if resume and quarter in prior_ok and target.exists():
            continue

        if target.exists() and remote_size and target.stat().st_size == remote_size:
            status = "skipped_existing"
            result = {
                "quarter": quarter,
                "url": url,
                "status": status,
                "size_mb": round(target.stat().st_size / 1024 / 1024, 2),
                "duration_s": 0,
                "timestamp": datetime.utcnow().isoformat(),
            }
            logs.append(result)
            results.append(result)
            continue

        if dry_run:
            result = {
                "quarter": quarter,
                "url": url,
                "status": "dry_run",
                "size_mb": round((remote_size or 0) / 1024 / 1024, 2),
                "duration_s": 0,
                "timestamp": datetime.utcnow().isoformat(),
            }
            logs.append(result)
            results.append(result)
            continue

        err = None
        start = time.time()
        for attempt in range(1, RETRY_COUNT + 1):
            tmp = target.with_suffix(target.suffix + ".tmp")
            try:
                with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
                    r.raise_for_status()
                    total = int(r.headers.get("Content-Length", 0))
                    with open(tmp, "wb") as f, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc=quarter,
                        ncols=110,
                    ) as pbar:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if _STOP:
                                raise KeyboardInterrupt
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                if remote_size and tmp.stat().st_size != remote_size:
                    raise IOError("size mismatch after download")
                tmp.replace(target)
                err = None
                break
            except Exception as ex:
                err = ex
                tmp.unlink(missing_ok=True)
                if attempt < RETRY_COUNT:
                    time.sleep(RETRY_BACKOFF[min(attempt - 1, len(RETRY_BACKOFF) - 1)])
        duration = round(time.time() - start, 2)
        status = "ok" if err is None else f"failed: {err}"
        result = {
            "quarter": quarter,
            "url": url,
            "status": status,
            "size_mb": round((target.stat().st_size if target.exists() else (remote_size or 0)) / 1024 / 1024, 2),
            "duration_s": duration,
            "timestamp": datetime.utcnow().isoformat(),
        }
        logs.append(result)
        results.append(result)
        _write_json(DOWNLOAD_LOG, logs)
        logger.info("download %s -> %s", quarter, status)
        if not _STOP:
            time.sleep(POLITE_DELAY)

    _write_json(DOWNLOAD_LOG, logs)
    _print_download_summary(results)
    return results


def _print_download_summary(rows: list[dict[str, Any]]) -> None:
    table = Table(title="Download Summary")
    for col in ["Quarter", "Status", "Size (MB)", "Duration (s)"]:
        table.add_column(col)
    for r in rows[-20:]:
        table.add_row(r["quarter"], r["status"], str(r["size_mb"]), str(r["duration_s"]))
    console.print(table)


def _quarter_from_filename(name: str) -> str | None:
    m = re.match(r"(?:aers|faers)_ascii_(\d{4})[Qq]([1-4])\.zip$", name)
    return f"{m.group(1)}Q{m.group(2)}" if m else None


def extract_all() -> list[dict[str, Any]]:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for zip_path in sorted(RAW_DIR.glob("*_ascii_*.zip")):
        if _STOP:
            break
        quarter = _quarter_from_filename(zip_path.name)
        if not quarter:
            continue
        out_dir = EXTRACT_DIR / quarter
        era = "aers" if zip_path.name.startswith("aers_") else "faers"
        demo_exists = list(out_dir.rglob("DEMO*.txt"))
        if demo_exists:
            results.append({"quarter": quarter, "era": era, "status": "skipped_existing"})
            continue
        try:
            with ZipFile(zip_path, "r") as zf:
                zf.extractall(out_dir)
            txt_files = list(out_dir.rglob("*.txt"))
            narr_files = [p for p in txt_files if p.name.upper().startswith("NARR")]
            narr_rows = _count_rows(narr_files[0]) if narr_files else 0
            results.append({
                "quarter": quarter,
                "era": era,
                "files_found": len(txt_files),
                "has_narr": bool(narr_files),
                "narr_row_estimate": narr_rows,
                "status": "ok",
            })
        except BadZipFile:
            logger.error("Bad zip: %s", zip_path)
            results.append({"quarter": quarter, "era": era, "status": "bad_zip"})
    return results


def _find_table_file(qdir: Path, table: str) -> Path | None:
    matches = list(qdir.rglob(f"{table}*.txt"))
    return matches[0] if matches else None


def _count_rows(path: Path) -> int:
    with path.open("r", encoding="latin1", errors="ignore") as f:
        n = -1
        for n, _ in enumerate(f):
            pass
    return max(0, n)


def validate_all() -> list[dict[str, Any]]:
    rows = []
    for qdir in sorted([p for p in EXTRACT_DIR.iterdir() if p.is_dir()]):
        if _STOP:
            break
        quarter = qdir.name
        year = int(quarter[:4])
        era = "aers" if year < 2012 or (year == 2012 and quarter.endswith("Q1") or quarter.endswith("Q2") or quarter.endswith("Q3")) else "faers"

        files = {t: _find_table_file(qdir, t) for t in EXPECTED_TABLES}
        missing = [k for k, v in files.items() if v is None]
        columns = {}
        counts = {}
        for t, p in files.items():
            if p is None:
                continue
            try:
                sample = pd.read_csv(p, sep="$", encoding="latin1", nrows=10, on_bad_lines="skip")
                columns[t] = list(sample.columns)
                counts[t] = _count_rows(p)
            except Exception as e:
                columns[t] = [f"read_error:{e}"]
                counts[t] = 0

        demo_cols = set(c.lower() for c in columns.get("DEMO", []))
        schema_flags = []
        if "primaryid" not in demo_cols:
            schema_flags.append("DEMO missing primaryid")
        if "caseid" not in demo_cols:
            schema_flags.append("DEMO missing caseid")
        key_integrity = {
            "demo_rows": counts.get("DEMO", 0),
            "missing_primaryid_rows": None,
            "missing_caseid_rows": None,
            "duplicate_primaryid_rows": None,
            "duplicate_caseid_rows": None,
            "primaryid_unique_ratio": None,
            "caseid_unique_ratio": None,
        }
        demo_path = files.get("DEMO")
        if demo_path and "primaryid" in demo_cols and "caseid" in demo_cols:
            try:
                demo_df = pd.read_csv(demo_path, sep="$", encoding="latin1", on_bad_lines="skip", dtype=str)
                pid = demo_df["primaryid"].fillna("").str.strip()
                cid = demo_df["caseid"].fillna("").str.strip()
                miss_pid = int((pid == "").sum())
                miss_cid = int((cid == "").sum())
                dup_pid = int(pid.duplicated(keep=False).sum())
                dup_cid = int(cid.duplicated(keep=False).sum())
                pid_unique_ratio = float(pid.nunique(dropna=True) / len(pid)) if len(pid) else 0.0
                cid_unique_ratio = float(cid.nunique(dropna=True) / len(cid)) if len(cid) else 0.0
                key_integrity.update(
                    {
                        "missing_primaryid_rows": miss_pid,
                        "missing_caseid_rows": miss_cid,
                        "duplicate_primaryid_rows": dup_pid,
                        "duplicate_caseid_rows": dup_cid,
                        "primaryid_unique_ratio": round(pid_unique_ratio, 6),
                        "caseid_unique_ratio": round(cid_unique_ratio, 6),
                    }
                )
                if miss_pid > 0:
                    schema_flags.append(f"DEMO missing primaryid rows: {miss_pid}")
                if miss_cid > 0:
                    schema_flags.append(f"DEMO missing caseid rows: {miss_cid}")
            except Exception as e:
                schema_flags.append(f"DEMO key integrity read error: {e}")
        narr_pct = (counts.get("NARR", 0) / counts.get("DEMO", 1)) if counts.get("DEMO", 0) else 0.0
        low_narr = narr_pct < 0.1

        rows.append({
            "quarter": quarter,
            "era": era,
            "missing_files": missing,
            "columns": columns,
            "counts": counts,
            "key_integrity": key_integrity,
            "narr_pct": round(narr_pct, 4),
            "low_narr": low_narr,
            "schema_flags": schema_flags,
            "status": "ok" if not missing and not schema_flags else "warning",
        })

    _write_json(VALIDATION_REPORT, rows)
    console.print(f"Validation complete: {len(rows)} quarters")
    return rows


def status_dashboard() -> None:
    validations = {r["quarter"]: r for r in _read_json_list(VALIDATION_REPORT)}
    downloads = _read_json_list(DOWNLOAD_LOG)
    downloaded_ok = {d["quarter"] for d in downloads if d.get("status") in {"ok", "skipped_existing"}}

    table = Table(title="FAERS Status Dashboard")
    for c in ["Quarter", "Era", "Downloaded", "Extracted", "DEMO rows", "NARR rows", "NARR%", "Status"]:
        table.add_column(c)

    total_demo = 0
    total_narr = 0
    for q in sorted({p.name for p in EXTRACT_DIR.glob("*Q*")} | set(downloaded_ok)):
        vr = validations.get(q, {})
        year = int(q[:4]) if q[:4].isdigit() else 0
        qnum = int(q[-1]) if q[-1].isdigit() else 0
        era = "AERS" if year < 2012 or (year == 2012 and qnum <= 3) else "FAERS"
        dl = "yes" if q in downloaded_ok else "no"
        ex = "yes" if (EXTRACT_DIR / q).exists() else "no"
        demo = vr.get("counts", {}).get("DEMO", 0)
        narr = vr.get("counts", {}).get("NARR", 0)
        pct = (narr / demo * 100) if demo else 0
        total_demo += demo
        total_narr += narr
        status = "complete" if dl == "yes" and ex == "yes" and vr else "partial" if dl == "yes" or ex == "yes" else "missing"
        color = "green" if status == "complete" else "yellow" if status == "partial" else "red"
        table.add_row(q, era, dl, ex, str(demo), str(narr), f"{pct:.1f}%", f"[{color}]{status}[/{color}]")

    raw_size = _dir_size_gb(RAW_DIR)
    total_size = _dir_size_gb(Path("data"))
    console.print(table)
    console.print(
        f"Totals: cases={total_demo:,} | with_narr={total_narr:,} | raw={raw_size:.2f} GB | total={total_size:.2f} GB"
    )


def _dir_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return size / (1024**3)


def estimate_storage(urls: list[dict[str, Any]]) -> None:
    total = len(urls)
    downloaded = 0
    remain_bytes = 0
    for u in urls:
        target = RAW_DIR / u["filename"]
        remote = _get_remote_size(u["url"]) or int((u.get("size_mb") or 0) * 1024 * 1024)
        if target.exists() and remote and target.stat().st_size == remote:
            downloaded += 1
        else:
            remain_bytes += remote

    remain_gb = remain_bytes / (1024**3)
    unzipped_gb = remain_gb * 6

    def h_at_mbps(mbps: int) -> float:
        bps = mbps * 1_000_000 / 8
        return (remain_bytes / bps) / 3600 if bps else 0.0

    console.print(f"Total files to download: {total}")
    console.print(f"Already downloaded: {downloaded}")
    console.print(f"Remaining download size: {remain_gb:.2f} GB")
    console.print(f"Estimated unzipped size: {unzipped_gb:.2f} GB")
    console.print(f"10 Mbps: {h_at_mbps(10):.2f} hours | 50 Mbps: {h_at_mbps(50):.2f} hours | 100 Mbps: {h_at_mbps(100):.2f} hours")


def print_final_box() -> None:
    vals = _read_json_list(VALIDATION_REPORT)
    downloads = _read_json_list(DOWNLOAD_LOG)
    q_ok = len({d["quarter"] for d in downloads if d.get("status") in {"ok", "skipped_existing"}})
    q_total = len(discover_urls())
    total_cases = sum(v.get("counts", {}).get("DEMO", 0) for v in vals)
    total_narr = sum(v.get("counts", {}).get("NARR", 0) for v in vals)
    failed = len([d for d in downloads if str(d.get("status", "")).startswith("failed")])
    raw_size = _dir_size_gb(RAW_DIR)
    total_size = _dir_size_gb(Path("data"))

    box = [
        "╔══════════════════════════════════════════╗",
        "║         FAERS Collection Complete        ║",
        "╠══════════════════════════════════════════╣",
        f"║  Quarters downloaded : {q_ok}/{q_total}".ljust(43) + "║",
        f"║  Total cases         : ~{total_cases:,}".ljust(43) + "║",
        f"║  Cases with NARR     : ~{total_narr:,}".ljust(43) + "║",
        f"║  Disk usage (raw)    : {raw_size:.1f} GB".ljust(43) + "║",
        f"║  Disk usage (total)  : {total_size:.1f} GB".ljust(43) + "║",
        f"║  Failed quarters     : {failed}".ljust(43) + "║",
        "╚══════════════════════════════════════════╝",
    ]
    console.print("\n".join(box))


def main() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    setup_logging()
    acquire_lock()
    try:
        parser = argparse.ArgumentParser(description="Collect FAERS/AERS quarterly ASCII files")
        parser.add_argument("--discover", action="store_true")
        parser.add_argument("--estimate", action="store_true")
        parser.add_argument("--download", action="store_true")
        parser.add_argument("--extract", action="store_true")
        parser.add_argument("--validate", action="store_true")
        parser.add_argument("--status", action="store_true")
        parser.add_argument("--all", action="store_true")
        parser.add_argument("--from", dest="start_year", type=int)
        parser.add_argument("--to", dest="end_year", type=int)
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--resume", action="store_true")
        args = parser.parse_args()

        urls = discover_urls()
        urls = _filter_range(urls, args.start_year, args.end_year)

        if args.discover:
            console.print_json(data=urls)
        if args.estimate:
            estimate_storage(urls)
        if args.download:
            download_all(urls, dry_run=args.dry_run, resume=args.resume)
        if args.extract:
            extract_all()
        if args.validate:
            validate_all()
        if args.status:
            status_dashboard()
        if args.all:
            estimate_storage(urls)
            download_all(urls, dry_run=args.dry_run, resume=args.resume)
            extract_all()
            validate_all()
            status_dashboard()
            print_final_box()
        if not any([args.discover, args.estimate, args.download, args.extract, args.validate, args.status, args.all]):
            parser.print_help()
    finally:
        release_lock()


if __name__ == "__main__":
    main()

"""
requirements.txt
requests
beautifulsoup4
pandas
tqdm
rich
"""
