from __future__ import annotations
from pathlib import Path
import os
import json
import joblib
import numpy as np
import pandas as pd

import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)

# Optional boosted models
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None


from sqlalchemy import create_engine, text

from pipelines.common.cmapss_fd001 import (
    load_cmapss_txt,
    add_rul_label,
    build_features,
    train_val_split_by_engine,
)


def compute_baseline_stats(
    X_train: np.ndarray, feature_cols: list[str], run_id: str, window: int
):
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
                "UPDATE model_registry SET is_active = FALSE WHERE name = :name AND stage = :stage AND is_active = TRUE"
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


def score(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data" / "raw"
    train_path = data_dir / "train_FD001.txt"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}. Put FD001 files in data/raw/")

    # MLflow
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("cmapss_rul_fd001_benchmark")

    # Data + features
    window = 10
    raw = load_cmapss_txt(train_path)
    labeled = add_rul_label(raw)
    feat_df, feature_cols = build_features(labeled, window=window)
    feat_df["RUL"] = labeled["RUL"].values

    train_df, val_df = train_val_split_by_engine(feat_df, val_frac=0.2, seed=42)

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["RUL"].to_numpy(dtype=float)
    X_val = val_df[feature_cols].to_numpy(dtype=float)
    y_val = val_df["RUL"].to_numpy(dtype=float)

    candidates = [
        ("ridge", Ridge(alpha=1.0, random_state=42)),
        ("rf", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)),
        ("gbr", GradientBoostingRegressor(random_state=42)),
        ("hgb", HistGradientBoostingRegressor(random_state=42)),
    ]

    if XGBRegressor is not None:
        candidates.append(
            (
                "xgb",
                XGBRegressor(
                    n_estimators=2000,
                    learning_rate=0.03,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            )
        )

    if LGBMRegressor is not None:
        candidates.append(
            (
                "lgbm",
                LGBMRegressor(
                    n_estimators=5000,
                    learning_rate=0.03,
                    num_leaves=63,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        )

    results = []
    best = None  # (rmse, run_id, model_key)

    artifacts_dir = repo_root / "artifacts" / "models"
    runs_dir = artifacts_dir / "runs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    for model_key, est in candidates:
        with mlflow.start_run(run_name=f"bench_{model_key}_w{window}"):
            mlflow.log_param("dataset", "FD001")
            mlflow.log_param("rolling_window", window)
            mlflow.log_param("model_key", model_key)
            mlflow.log_param("estimator", est.__class__.__name__)

            # Log hyperparams in a generic way
            for k, v in est.get_params().items():
                if isinstance(v, (str, int, float, bool)) and len(str(k)) < 40:
                    mlflow.log_param(f"hp_{k}", v)

            run_id = mlflow.active_run().info.run_id

            if model_key == "xgb":
                est.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                est.fit(X_train, y_train)
            pred = est.predict(X_val)

            mae, rmse, r2 = score(y_val, pred)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)

            # Save run-specific model bundle for promotion
            bundle = {"model": est, "feature_names": feature_cols}
            run_bundle_path = runs_dir / f"{run_id}.pkl"
            joblib.dump(bundle, run_bundle_path)
            mlflow.log_artifact(str(run_bundle_path), artifact_path="model_bundle")

            # Save baseline stats tied to this run (for monitoring)
            stats = compute_baseline_stats(
                X_train, feature_cols, run_id=run_id, window=window
            )
            stats_path = artifacts_dir / "baseline_stats.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            mlflow.log_artifact(str(stats_path), artifact_path="model")

            results.append(
                {
                    "model_key": model_key,
                    "run_id": run_id,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                }
            )

            if best is None or rmse < best[0]:
                best = (rmse, run_id, model_key)

    # Register best as active dev
    if best:
        rmse, run_id, model_key = best
        # Note: store container-visible path
        register_model(
            run_id=run_id,
            artifact_path=f"/artifacts/models/runs/{run_id}.pkl",
            stage="dev",
            name="cmapss_fd001_rul",
            notes=f"best_benchmark={model_key} rmse={rmse:.4f}",
        )

    # Save summary artifact
    summary_path = repo_root / "artifacts" / "benchmark_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"best": best, "results": results}, f, indent=2)

    print("Benchmark complete. Results:")
    for r in sorted(results, key=lambda x: x["RMSE"]):
        print(r)
    if best:
        print(f"Best model: {best[2]} (run_id={best[1]}) RMSE={best[0]:.4f}")
        print("Registered as active dev in model_registry.")
        print("Promote via API when ready: POST /model/promote")


if __name__ == "__main__":
    main()
