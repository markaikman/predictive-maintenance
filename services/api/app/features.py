from __future__ import annotations

import pandas as pd


def build_features_for_engine(
    df_engine: pd.DataFrame, window: int = 10
) -> tuple[pd.DataFrame, list[str]]:
    """
    Given raw per-cycle rows for ONE engine (must include settings_1..3 and s1..s21 and cycle),
    return a dataframe of features per cycle and the feature column order.
    """
    base_cols = [f"setting_{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
    out = df_engine[["cycle"] + base_cols].copy()

    for c in [f"s{i}" for i in range(1, 22)]:
        out[f"{c}_rm{window}"] = out[c].rolling(window=window, min_periods=1).mean()
        out[f"{c}_rs{window}"] = (
            out[c].rolling(window=window, min_periods=1).std().fillna(0.0)
        )

    feature_cols = [c for c in out.columns if c != "cycle"]
    return out, feature_cols
