"""
FAERSDataset — chunked PyTorch Dataset for large parquet splits.

Loads parquet in row-group chunks so 36M-row train fits on any RAM.
Each item: (text_input, smiles, label).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from pharmviginet.config import (
    TRAIN_PARQUET, VAL_PARQUET, TEST_PARQUET,
    PUBMEDBERT, TEXT_MAX_LEN, BATCH_SIZE,
)

_TEXT_COLS = ["text_input", "smiles", "label"]


class FAERSDataset(Dataset):
    """
    Streams parquet row-groups into memory one at a time.
    Call .load_chunk(i) before indexing — or use FAERSIterDataset for auto-streaming.
    """

    def __init__(self, path: Path, tokenizer, max_len: int = TEXT_MAX_LEN,
                 sample_n: Optional[int] = None, seed: int = 42):
        self.path = path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pf = pq.ParquetFile(path)
        self.n_groups = self.pf.num_row_groups
        self._texts: list[str] = []
        self._smiles: list[str] = []
        self._labels: list[int] = []
        self.sample_n = sample_n
        self.seed = seed

    def load_chunk(self, group_idx: int) -> None:
        tbl = self.pf.read_row_group(group_idx, columns=_TEXT_COLS)
        self._texts  = tbl["text_input"].to_pylist()
        self._smiles = tbl["smiles"].to_pylist()
        self._labels = tbl["label"].to_pylist()

    def load_all(self) -> "FAERSDataset":
        """Load full dataset into memory (use only for val/test)."""
        import pyarrow as pa
        tbl = pq.read_table(self.path, columns=_TEXT_COLS)
        self._texts  = tbl["text_input"].to_pylist()
        self._smiles = tbl["smiles"].to_pylist()
        self._labels = tbl["label"].to_pylist()
        if self.sample_n and self.sample_n < len(self._texts):
            rng = random.Random(self.seed)
            idx = rng.sample(range(len(self._texts)), self.sample_n)
            self._texts  = [self._texts[i]  for i in idx]
            self._smiles = [self._smiles[i] for i in idx]
            self._labels = [self._labels[i] for i in idx]
        return self

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self._texts[idx] or "",
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "smiles":         self._smiles[idx] or "[UNK-MOL]",
            "label":          torch.tensor(int(self._labels[idx] or 0), dtype=torch.float),
        }


def make_loader(path: Path, tokenizer, batch_size: int = BATCH_SIZE,
                sample_n: Optional[int] = None, shuffle: bool = False) -> DataLoader:
    ds = FAERSDataset(path, tokenizer, sample_n=sample_n).load_all()
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)
