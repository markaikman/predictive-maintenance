from __future__ import annotations

from pathlib import Path
import re
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipelines.common.cmapss_fd001 import load_cmapss_txt, build_features


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def infer_window_from_feature_names(feature_names: list[str], default: int = 30) -> int:
    # Look for patterns like s3_rm30, s7_rs10, etc.
    wins = []
    for n in feature_names:
        m = re.search(r"_rm(\d+)$", n)
        if m:
            wins.append(int(m.group(1)))
        m = re.search(r"_rs(\d+)$", n)
        if m:
            wins.append(int(m.group(1)))
    return max(wins) if wins else default


def load_test_labels(repo_root: Path) -> pd.DataFrame:
    test_path = repo_root / "data" / "raw" / "test_FD001.txt"
    rul_path = repo_root / "data" / "raw" / "RUL_FD001.txt"

    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}")
    if not rul_path.exists():
        raise FileNotFoundError(f"Missing {rul_path}")

    test_raw = load_cmapss_txt(test_path)

    # RUL file is one integer per engine (in engine_id order: 1..N)
    rul_vals = pd.read_csv(rul_path, header=None).iloc[:, 0].astype(int).to_numpy()
    n_engines = int(test_raw["engine_id"].nunique())
    if len(rul_vals) != n_engines:
        raise ValueError(
            f"RUL_FD001.txt length ({len(rul_vals)}) != num engines in test ({n_engines})"
        )

    rul_df = pd.DataFrame(
        {"engine_id": np.arange(1, n_engines + 1, dtype=int), "rul_last": rul_vals}
    )

    max_cycle = (
        test_raw.groupby("engine_id")["cycle"].max().rename("max_cycle").reset_index()
    )

    df = test_raw.merge(max_cycle, on="engine_id", how="left").merge(
        rul_df, on="engine_id", how="left"
    )

    # True RUL for each cycle:
    # RUL(row) = RUL_last(engine) + (max_cycle(engine) - cycle(row))
    df["RUL_true"] = df["rul_last"] + (df["max_cycle"] - df["cycle"])

    return df


def main():
    repo_root = Path(__file__).resolve().parents[2]

    # Load training distribution for comparison
    train_path = repo_root / "data" / "raw" / "train_FD001.txt"
    train_raw = load_cmapss_txt(train_path)
    train_labeled = train_raw.copy()
    # compute full train RUL (since train includes full life)
    max_cycle_train = (
        train_labeled.groupby("engine_id")["cycle"]
        .max()
        .rename("max_cycle")
        .reset_index()
    )
    train_labeled = train_labeled.merge(max_cycle_train, on="engine_id", how="left")
    train_labeled["RUL_train"] = train_labeled["max_cycle"] - train_labeled["cycle"]

    print("\n--- TRAIN RUL Distribution ---")
    print("Train RUL mean :", float(train_labeled["RUL_train"].mean()))
    print("Train RUL std  :", float(train_labeled["RUL_train"].std()))
    print("Train RUL max  :", float(train_labeled["RUL_train"].max()))
    print("Train RUL min  :", float(train_labeled["RUL_train"].min()))
    out_dir = repo_root / "artifacts" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load production bundle (sklearn/lightgbm bundle)
    prod_path = repo_root / "artifacts" / "models" / "production.pkl"
    if not prod_path.exists():
        raise FileNotFoundError(f"Missing production bundle: {prod_path}")

    bundle = joblib.load(prod_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    test_df = load_test_labels(repo_root)

    # Diagnostic: compare RUL distributions
    print("\n--- RUL Distribution Diagnostics ---")
    print("Test RUL mean :", float(test_df["RUL_true"].mean()))
    print("Test RUL std  :", float(test_df["RUL_true"].std()))
    print("Test RUL max  :", float(test_df["RUL_true"].max()))
    print("Test RUL min  :", float(test_df["RUL_true"].min()))

    # Build features using the same rolling window implied by feature names
    window = infer_window_from_feature_names(feature_names, default=30)
    feat_df, feature_cols = build_features(test_df, window=window)

    # Align predictions with the feature order expected by the prod model
    X = feat_df[feature_names].astype(float)
    y = test_df["RUL_true"].to_numpy(dtype=float)

    pred = model.predict(X)

    overall = metrics(y, pred)

    # Also evaluate only the LAST cycle per engine (most realistic operationally)
    idx_last = test_df.groupby("engine_id")["cycle"].idxmax().to_numpy()
    y_last = test_df.loc[idx_last, "RUL_true"].to_numpy(dtype=float)

    train_mean_rul = float(train_labeled["RUL_train"].mean())
    pred_naive = np.full_like(y, train_mean_rul, dtype=float)
    naive_overall = metrics(y, pred_naive)

    pred_naive_last = np.full_like(y_last, train_mean_rul, dtype=float)
    naive_last = metrics(y_last, pred_naive_last)

    print("\n=== Naive baseline (predict train mean RUL) ===")
    print("All rows :", naive_overall)
    print("Last only:", naive_last)

    X_last = feat_df.loc[idx_last, feature_names].astype(float)
    pred_last = model.predict(X_last)

    last_only = metrics(y_last, pred_last)

    report = {
        "dataset": "FD001",
        "model_artifact": str(prod_path),
        "window_inferred": window,
        "n_rows_test": int(len(test_df)),
        "n_engines_test": int(test_df["engine_id"].nunique()),
        "overall_all_rows": overall,
        "last_cycle_per_engine": last_only,
    }

    print("\n=== FD001 TEST EVAL (prod bundle) ===")
    print("All rows :", report["overall_all_rows"])
    print("Last only:", report["last_cycle_per_engine"])
    print("window inferred:", window)

    out_path = out_dir / "fd001_test_eval_prod.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
