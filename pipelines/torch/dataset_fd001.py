from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class SeqConfig:
    seq_len: int = 30
    batch_size: int = 256
    num_workers: int = 0  # keep 0 on Windows for fewer headaches
    shuffle_train: bool = True


class SequenceDataset(Dataset):
    """
    Each item is (X_seq, y) where:
      X_seq: (seq_len, n_features)
      y: scalar (RUL at final timestep)
    """

    def __init__(self, X_seq: np.ndarray, y: np.ndarray):
        assert X_seq.ndim == 3
        assert y.ndim == 1
        assert X_seq.shape[0] == y.shape[0]
        self.X = torch.from_numpy(X_seq).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def _fit_standardizer(X2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    X2 shape: (N, F)
    Returns mean, std (with std floor to avoid div-by-zero)
    """
    mean = X2.mean(axis=0)
    std = X2.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _apply_standardizer(
    X2: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return (X2 - mean) / std


def make_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sequences per engine without leakage across engines.
    Assumes df has columns: engine_id, cycle, RUL + feature_cols.
    Returns:
      X_seq: (num_samples, seq_len, num_features)
      y: (num_samples,)
    """
    X_list: list[np.ndarray] = []
    y_list: list[float] = []

    # Ensure sorted within engine
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    for eng_id, g in df.groupby("engine_id", sort=False):
        g = g.reset_index(drop=True)
        X_eng = g[feature_cols].to_numpy(dtype=np.float32)
        y_eng = g["RUL"].to_numpy(dtype=np.float32)

        if len(g) < seq_len:
            continue

        # sliding windows: last timestep label
        for end in range(seq_len - 1, len(g)):
            start = end - (seq_len - 1)
            X_list.append(X_eng[start : end + 1])
            y_list.append(float(y_eng[end]))

    X_seq = (
        np.stack(X_list, axis=0)
        if X_list
        else np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32)
    )
    y = (
        np.array(y_list, dtype=np.float32)
        if y_list
        else np.zeros((0,), dtype=np.float32)
    )
    return X_seq, y


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: SeqConfig,
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Fits a standardizer on TRAIN only (flattened timesteps), applies to train/val,
    creates sequences, returns loaders + scaler metadata.
    """
    # Fit scaler on training rows (not sequences) to avoid leaking val stats
    X_train_2d = train_df[feature_cols].to_numpy(dtype=np.float32)
    mean, std = _fit_standardizer(X_train_2d)

    # Apply scaler to row-wise features, then build sequences
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()

    train_scaled.loc[:, feature_cols] = _apply_standardizer(
        train_scaled[feature_cols].to_numpy(dtype=np.float32), mean, std
    )
    val_scaled.loc[:, feature_cols] = _apply_standardizer(
        val_scaled[feature_cols].to_numpy(dtype=np.float32), mean, std
    )

    Xtr, ytr = make_sequences(train_scaled, feature_cols, cfg.seq_len)
    Xva, yva = make_sequences(val_scaled, feature_cols, cfg.seq_len)

    train_ds = SequenceDataset(Xtr, ytr)
    val_ds = SequenceDataset(Xva, yva)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_train,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    meta = {
        "scaler_mean": mean.astype(float).tolist(),
        "scaler_std": std.astype(float).tolist(),
        "seq_len": cfg.seq_len,
        "feature_cols": feature_cols,
    }
    return train_loader, val_loader, meta
