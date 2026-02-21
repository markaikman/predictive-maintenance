from __future__ import annotations

from pathlib import Path
import os
import json
import joblib
import numpy as np
import pandas as pd

import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sqlalchemy import create_engine, text

from pipelines.common.cmapss_fd001 import (
    load_cmapss_txt,
    add_rul_label,
    build_features,
    train_val_split_by_engine,
)

# Optional deps
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def score(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r = rmse(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return float(mae), float(r), float(r2)


def compute_baseline_stats(
    X_train: np.ndarray, feature_cols: list[str], run_id: str, window: int
) -> dict:
    baseline = {}
    X_train_df = pd.DataFrame(X_train, columns=feature_cols)
    for c in feature_cols:
        s = X_train_df[c].astype(float)
        baseline[c] = {
            "p05": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
        }
    return {
        "dataset": "FD001",
        "rolling_window": window,
        "feature_names": feature_cols,
        "baseline": baseline,
        "run_id": run_id,
    }


def register_model(
    run_id: str, artifact_path: str, stage: str, name: str, notes: str | None = None
):
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL env var required to register model")
    eng = create_engine(db_url)
    with eng.begin() as conn:
        conn.execute(
            text(
                "UPDATE model_registry SET is_active=FALSE WHERE name=:name AND stage=:stage AND is_active=TRUE"
            ),
            {"name": name, "stage": stage},
        )
        conn.execute(
            text(
                """
                INSERT INTO model_registry (name, stage, run_id, artifact_path, notes, is_active)
                VALUES (:name, :stage, :run_id, :artifact_path, :notes, TRUE)
            """
            ),
            {
                "name": name,
                "stage": stage,
                "run_id": run_id,
                "artifact_path": artifact_path,
                "notes": notes,
            },
        )


def sample_lgbm_params(rng: np.random.Generator) -> dict:
    # Reasonable search space for tabular regression
    return {
        "n_estimators": 20000,  # large; early stopping will pick best iteration
        "learning_rate": float(rng.choice([0.01, 0.02, 0.03, 0.05])),
        "num_leaves": int(rng.choice([31, 63, 127, 255])),
        "min_child_samples": int(rng.choice([10, 20, 40, 80])),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "reg_alpha": float(rng.choice([0.0, 1e-3, 1e-2, 1e-1])),
        "reg_lambda": float(rng.choice([0.0, 1e-2, 1e-1, 1.0, 5.0])),
        "max_depth": int(rng.choice([-1, 6, 10, 14])),
    }


def sample_xgb_params(rng: np.random.Generator) -> dict:
    return {
        "n_estimators": 20000,
        "learning_rate": float(rng.choice([0.01, 0.02, 0.03, 0.05])),
        "max_depth": int(rng.choice([3, 4, 5, 6, 8])),
        "min_child_weight": float(rng.choice([1, 2, 5, 10])),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "reg_alpha": float(rng.choice([0.0, 1e-3, 1e-2, 1e-1])),
        "reg_lambda": float(rng.choice([0.1, 0.5, 1.0, 2.0, 5.0])),
        "gamma": float(rng.choice([0.0, 0.1, 0.2, 0.5])),
    }


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data" / "raw"
    train_path = data_dir / "train_FD001.txt"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}. Put FD001 files in data/raw/")

    # Settings (start narrow; validate wider after we find a winner)
    window = 30
    seed = 42
    val_frac = 0.2
    n_trials = 40
    model_family = os.getenv("TUNE_MODEL", "lgbm").lower()  # "lgbm" or "xgb"

    if model_family == "lgbm" and LGBMRegressor is None:
        raise RuntimeError("lightgbm is not installed. Run: pip install lightgbm")
    if model_family == "xgb" and XGBRegressor is None:
        raise RuntimeError("xgboost is not installed. Run: pip install xgboost")

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("cmapss_rul_fd001_tuning")

    # Artifacts dirs
    artifacts_dir = repo_root / "artifacts" / "models"
    runs_dir = artifacts_dir / "runs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Data prep
    raw = load_cmapss_txt(train_path)
    labeled = add_rul_label(raw)
    feat_df, feature_cols = build_features(labeled, window=window)
    feat_df["RUL"] = labeled["RUL"].values

    train_df, val_df = train_val_split_by_engine(feat_df, val_frac=val_frac, seed=seed)

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["RUL"].to_numpy(dtype=float)
    X_val = val_df[feature_cols].to_numpy(dtype=float)
    y_val = val_df["RUL"].to_numpy(dtype=float)

    rng = np.random.default_rng(1234)

    best = None  # (rmse, run_id, params)

    for t in range(n_trials):
        if model_family == "lgbm":
            params = sample_lgbm_params(rng)
            est = LGBMRegressor(
                **params,
                random_state=42,
                n_jobs=-1,
            )
        else:
            params = sample_xgb_params(rng)
            est = XGBRegressor(
                **params,
                random_state=42,
                n_jobs=-1,
                objective="reg:squarederror",
                tree_method="hist",
            )

        with mlflow.start_run(run_name=f"tune_{model_family}_w{window}_s{seed}_t{t+1}"):
            mlflow.log_param("dataset", "FD001")
            mlflow.log_param("rolling_window", window)
            mlflow.log_param("split_seed", seed)
            mlflow.log_param("val_frac_by_engine", val_frac)
            mlflow.log_param("model_family", model_family)
            for k, v in params.items():
                mlflow.log_param(f"hp_{k}", v)

            run_id = mlflow.active_run().info.run_id

            # Early stopping
            if model_family == "lgbm":
                est.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="rmse",
                    callbacks=[],  # keep quiet
                )
            else:
                est.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )

            pred = est.predict(X_val)
            mae, r, r2 = score(y_val, pred)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", r)
            mlflow.log_metric("R2", r2)

            # Save run bundle
            bundle = {"model": est, "feature_names": feature_cols}
            run_bundle_path = runs_dir / f"{run_id}.pkl"
            joblib.dump(bundle, run_bundle_path)
            mlflow.log_artifact(str(run_bundle_path), artifact_path="model_bundle")

            # Save baseline stats for this run
            stats = compute_baseline_stats(
                X_train, feature_cols, run_id=run_id, window=window
            )
            stats_path = artifacts_dir / f"baseline_stats_{run_id}.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            mlflow.log_artifact(str(stats_path), artifact_path="model")

            if best is None or r < best[0]:
                best = (r, run_id, params)

    if not best:
        raise RuntimeError("No tuning trials completed")

    best_rmse, best_run_id, best_params = best
    print(f"Best {model_family} trial: run_id={best_run_id} RMSE={best_rmse:.4f}")

    # Register best as dev
    if os.getenv("DATABASE_URL"):
        register_model(
            run_id=best_run_id,
            artifact_path=f"/artifacts/models/runs/{best_run_id}.pkl",
            stage="dev",
            name="cmapss_fd001_rul",
            notes=f"tuned_{model_family} rmse={best_rmse:.4f} window={window} seed={seed}",
        )
        print("Registered best tuned model as active dev.")
    else:
        print("DATABASE_URL not set; skipping dev registration.")

    print("Promote via API when ready: POST /model/promote")


if __name__ == "__main__":
    main()
