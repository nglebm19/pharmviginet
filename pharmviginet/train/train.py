#!/usr/bin/env python3
"""
Training orchestrator — runs text model then mol model sequentially.

Usage:
    python -m pharmviginet.train.train                        # full run
    python -m pharmviginet.train.train --sample 500000        # quick run
    python -m pharmviginet.train.train --only text            # text only
    python -m pharmviginet.train.train --only mol             # mol only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Subsample N rows per split (default: full)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override MAX_EPOCHS from config")
    parser.add_argument("--only", choices=["text", "mol"], default=None,
                        help="Run only one model (default: both)")
    args = parser.parse_args()

    base = [sys.executable, "-m"]
    extra = []
    if args.sample:
        extra += ["--sample", str(args.sample)]
    if args.epochs:
        extra += ["--epochs", str(args.epochs)]

    t0 = time.time()

    if args.only in (None, "text"):
        run(base + ["pharmviginet.models.text"] + extra)

    if args.only in (None, "mol"):
        run(base + ["pharmviginet.models.mol"] + extra)

    total = (time.time() - t0) / 60
    print(f"\nTotal training time: {total:.1f} min")
    print("Checkpoints saved to model_checkpoints/")
    print("Results saved to data/logs/")


if __name__ == "__main__":
    main()
