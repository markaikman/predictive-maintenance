from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np


def load_cmapss_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.iloc[:, -1].isna().all():
        df = df.iloc[:, :-1]
    cols = (
        ["engine_id", "cycle"]
        + [f"setting_{i}" for i in range(1, 4)]
        + [f"s{i}" for i in range(1, 22)]
    )
    if len(df.columns) != len(cols):
        raise ValueError(
            f"Unexpected column count {len(df.columns)} in {path}. Expected {len(cols)}."
        )
    df.columns = cols
    return df


def add_rul_label(train_df: pd.DataFrame) -> pd.DataFrame:
    max_cycle = train_df.groupby("engine_id")["cycle"].max().rename("max_cycle")
    df = train_df.merge(max_cycle, on="engine_id", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def build_features(
    df: pd.DataFrame, window: int = 10
) -> tuple[pd.DataFrame, list[str]]:
    base_cols = [f"setting_{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
    out = df[["engine_id", "cycle"] + base_cols].copy()

    for c in [f"s{i}" for i in range(1, 22)]:
        out[f"{c}_rm{window}"] = (
            df.groupby("engine_id")[c]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        out[f"{c}_rs{window}"] = (
            df.groupby("engine_id")[c]
            .rolling(window=window, min_periods=1)
            .std()
            .fillna(0.0)
            .reset_index(level=0, drop=True)
        )

    feature_cols = [c for c in out.columns if c not in ("engine_id", "cycle")]
    return out, feature_cols


def train_val_split_by_engine(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    engines = df["engine_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(engines)
    n_val = max(1, int(len(engines) * val_frac))
    val_engines = set(engines[:n_val])
    train = df[~df["engine_id"].isin(val_engines)].copy()
    val = df[df["engine_id"].isin(val_engines)].copy()
    return train, val
