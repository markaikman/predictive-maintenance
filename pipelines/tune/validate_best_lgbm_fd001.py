from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipelines.common.cmapss_fd001 import (
    load_cmapss_txt,
    add_rul_label,
    build_features,
    train_val_split_by_engine,
)


def score(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return mae, rmse, r2


def main():
    repo_root = Path(__file__).resolve().parents[2]
    train_path = repo_root / "data" / "raw" / "train_FD001.txt"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")

    # <<< Paste your best params here >>>
    BEST_PARAMS = {
        # example placeholders — replace with your actual best trial params
        "n_estimators": 20000,
        "learning_rate": 0.03,
        "num_leaves": 127,
        "min_child_samples": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "max_depth": -1,
    }

    window = 30
    val_frac = 0.2
    seeds = [42, 123, 999]

    raw = load_cmapss_txt(train_path)
    labeled = add_rul_label(raw)
    feat_df, feature_cols = build_features(labeled, window=window)
    feat_df["RUL"] = labeled["RUL"].values

    rows = []
    for seed in seeds:
        train_df, val_df = train_val_split_by_engine(
            feat_df, val_frac=val_frac, seed=seed
        )

        X_train = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df["RUL"].to_numpy(dtype=float)
        X_val = val_df[feature_cols].to_numpy(dtype=float)
        y_val = val_df["RUL"].to_numpy(dtype=float)

        model = LGBMRegressor(**BEST_PARAMS, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="rmse")

        pred = model.predict(X_val)
        mae, rmse, r2 = score(y_val, pred)

        rows.append({"seed": seed, "MAE": mae, "RMSE": rmse, "R2": r2})
        print(f"seed={seed}  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")

    df = pd.DataFrame(rows)
    print("\nSummary:")
    print(df.describe().loc[["mean", "std"]])

    out_dir = repo_root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "validate_best_lgbm_w30.csv"
    df.to_csv(out_path, index=False)
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
