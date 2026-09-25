#!/usr/bin/env python3
"""
Text model — PubMedBERT fine-tuned on FAERS text_input field.

Usage:
    python -m pharmviginet.models.text --sample 500000   # quick validation run
    python -m pharmviginet.models.text                   # full training
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from pharmviginet.config import (
    PUBMEDBERT, TEXT_MAX_LEN, BATCH_SIZE, LR, WEIGHT_DECAY,
    MAX_EPOCHS, POS_WEIGHT, LOGS, CKPT_DIR,
    TRAIN_PARQUET, VAL_PARQUET,
)
from pharmviginet.data.faers import make_loader
from pharmviginet.utils.metrics import compute_metrics, print_metrics, save_metrics


class TextModel(nn.Module):
    def __init__(self, model_name: str = PUBMEDBERT, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(cls).squeeze(-1)  # (B,)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(model, loader, optimizer, scheduler, criterion, device) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels    = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attn_mask)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(labels)
        n += len(labels)

    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_scores, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, attn_mask)
        scores = torch.sigmoid(logits).cpu().numpy()
        all_scores.extend(scores.tolist())
        all_labels.extend(batch["label"].numpy().tolist())

    import numpy as np
    return compute_metrics(np.array(all_labels), np.array(all_scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Subsample N rows for quick test (default: full dataset)")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    print(f"Loading tokenizer: {PUBMEDBERT}")
    tokenizer = AutoTokenizer.from_pretrained(PUBMEDBERT)

    sample_val = min(50_000, args.sample) if args.sample else 50_000
    print(f"Loading data (train sample={args.sample}, val sample={sample_val}) …")
    train_loader = make_loader(TRAIN_PARQUET, tokenizer,
                               batch_size=args.batch_size,
                               sample_n=args.sample, shuffle=True)
    val_loader   = make_loader(VAL_PARQUET, tokenizer,
                               batch_size=args.batch_size * 2,
                               sample_n=sample_val)
    print(f"  train batches={len(train_loader)}, val batches={len(val_loader)}")

    print(f"Loading model: {PUBMEDBERT}")
    model = TextModel().to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(POS_WEIGHT, device=device)
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_auc, results = 0.0, {}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} | "
              f"val_AUC={val_metrics['auc']:.4f} | {elapsed:.0f}s")
        print_metrics(f"val/epoch{epoch}", val_metrics)

        results[f"epoch_{epoch}"] = {"train_loss": train_loss, **val_metrics}

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            ckpt = CKPT_DIR / "text_best.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"  → New best AUC {best_auc:.4f}, saved {ckpt}")

    save_metrics(LOGS / "text_results.json", results)
    print(f"\nBest val AUC: {best_auc:.4f}")
    print(f"Baseline to beat (ROR_train test): 0.8928")


if __name__ == "__main__":
    main()
