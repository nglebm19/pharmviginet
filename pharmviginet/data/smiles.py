"""
MolDataset — PyTorch Dataset for SMILES-only training (ChemBERTa).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader

from pharmviginet.config import MOL_MAX_LEN, BATCH_SIZE

_MOL_COLS = ["smiles", "label"]


class MolDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_len: int = MOL_MAX_LEN,
                 sample_n: Optional[int] = None, seed: int = 42):
        self.tokenizer = tokenizer
        self.max_len = max_len
        tbl = pq.read_table(path, columns=_MOL_COLS)
        smiles = tbl["smiles"].to_pylist()
        labels = tbl["label"].to_pylist()
        if sample_n and sample_n < len(smiles):
            rng = random.Random(seed)
            idx = rng.sample(range(len(smiles)), sample_n)
            smiles = [smiles[i] for i in idx]
            labels = [labels[i] for i in idx]
        self._smiles = smiles
        self._labels = labels

    def __len__(self) -> int:
        return len(self._smiles)

    def __getitem__(self, idx: int) -> dict:
        smi = self._smiles[idx] or "[UNK-MOL]"
        enc = self.tokenizer(
            smi,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(int(self._labels[idx] or 0), dtype=torch.float),
        }


def make_mol_loader(path: Path, tokenizer, batch_size: int = BATCH_SIZE,
                    sample_n: Optional[int] = None, shuffle: bool = False) -> DataLoader:
    ds = MolDataset(path, tokenizer, sample_n=sample_n)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=4, pin_memory=True)
