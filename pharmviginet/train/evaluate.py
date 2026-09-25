#!/usr/bin/env python3
"""
Evaluate trained text and mol models on val and test splits.

Usage:
    python -m pharmviginet.train.evaluate
    python -m pharmviginet.train.evaluate --split val
    python -m pharmviginet.train.evaluate --model text
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from pharmviginet.config import (
    PUBMEDBERT, CHEMBERT, CKPT_DIR, LOGS,
    VAL_PARQUET, TEST_PARQUET, BATCH_SIZE,
)
from pharmviginet.data.faers import make_loader
from pharmviginet.data.smiles import make_mol_loader
from pharmviginet.models.text import TextModel, evaluate as eval_text
from pharmviginet.models.mol import MolModel, evaluate as eval_mol
from pharmviginet.utils.metrics import compute_metrics, print_metrics, save_metrics
from transformers import AutoTokenizer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_model(model_type: str, split: str, device: torch.device) -> dict:
    path = VAL_PARQUET if split == "val" else TEST_PARQUET
    results = {}

    if model_type in ("text", "both"):
        ckpt = CKPT_DIR / "text_best.pt"
        if not ckpt.exists():
            print(f"  [text] {ckpt} not found — skipping")
        else:
            tokenizer = AutoTokenizer.from_pretrained(PUBMEDBERT)
            loader = make_loader(path, tokenizer, batch_size=BATCH_SIZE * 2,
                                 sample_n=100_000)
            model = TextModel().to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            metrics = eval_text(model, loader, device)
            print_metrics(f"text/{split}", metrics)
            results["text"] = metrics

    if model_type in ("mol", "both"):
        ckpt = CKPT_DIR / "mol_best.pt"
        if not ckpt.exists():
            print(f"  [mol] {ckpt} not found — skipping")
        else:
            tokenizer = AutoTokenizer.from_pretrained(CHEMBERT)
            loader = make_mol_loader(path, tokenizer, batch_size=BATCH_SIZE * 2,
                                     sample_n=100_000)
            model = MolModel().to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            metrics = eval_mol(model, loader, device)
            print_metrics(f"mol/{split}", metrics)
            results["mol"] = metrics

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    parser.add_argument("--model", choices=["text", "mol", "both"], default="both")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    all_results = {}
    splits = ["val", "test"] if args.split == "both" else [args.split]
    for split in splits:
        print(f"\n── {split.upper()} ──")
        all_results[split] = evaluate_model(args.model, split, device)

    out = LOGS / "eval_results.json"
    save_metrics(out, all_results)
    print(f"\nSaved → {out}")
    print(f"ROR_train baseline: AUC=0.8928 (test)")


if __name__ == "__main__":
    main()
