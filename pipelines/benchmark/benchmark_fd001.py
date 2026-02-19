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

from sqlalchemy import create_engine, text

from pipelines.common.cmapss_fd001 import (
    load_cmapss_txt,
    add_rul_label,
    build_features,
    train_val_split_by_engine,
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
    run_id: str,
    artifact_path: str,
    stage: str,
    name: str,
    notes: str | None = None,
) -> None:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL env var required to register model")

    eng = create_engine(db_url)
    with eng.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE model_registry
                SET is_active = FALSE
                WHERE name = :name AND stage = :stage AND is_active = TRUE
                """
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


def score(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def build_candidates() -> list[tuple[str, object]]:
    candidates: list[tuple[str, object]] = [
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

    return candidates


def cleanup_local_run_artifacts(
    runs_dir: Path, artifacts_dir: Path, keep_last: int = 50
):
    pkls = sorted(runs_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in pkls[keep_last:]:
        p.unlink(missing_ok=True)

    stats = sorted(
        artifacts_dir.glob("baseline_stats_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in stats[keep_last:]:
        p.unlink(missing_ok=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data" / "raw"
    train_path = data_dir / "train_FD001.txt"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}. Put FD001 files in data/raw/")

    # ---- MLflow ----
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("cmapss_rul_fd001_benchmark")

    # ---- Rigor settings ----
    windows = [5, 10, 20, 30]
    seeds = [42, 123, 999]
    val_frac = 0.2

    # ---- Artifact directories (defined ONCE) ----
    artifacts_dir = repo_root / "artifacts" / "models"
    runs_dir = artifacts_dir / "runs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load + label once (features built per window) ----
    raw = load_cmapss_txt(train_path)
    labeled = add_rul_label(raw)

    all_results: list[dict] = []
    candidates = build_candidates()

    for window in windows:
        feat_df, feature_cols = build_features(labeled, window=window)
        feat_df["RUL"] = labeled["RUL"].values

        for seed in seeds:
            train_df, val_df = train_val_split_by_engine(
                feat_df, val_frac=val_frac, seed=seed
            )

            X_train = train_df[feature_cols].to_numpy(dtype=float)
            y_train = train_df["RUL"].to_numpy(dtype=float)
            X_val = val_df[feature_cols].to_numpy(dtype=float)
            y_val = val_df["RUL"].to_numpy(dtype=float)

            for model_key, est in candidates:
                with mlflow.start_run(run_name=f"bench_{model_key}_w{window}_s{seed}"):
                    mlflow.log_param("dataset", "FD001")
                    mlflow.log_param("rolling_window", window)
                    mlflow.log_param("split_seed", seed)
                    mlflow.log_param("val_frac_by_engine", val_frac)
                    mlflow.log_param("model_key", model_key)
                    mlflow.log_param("estimator", est.__class__.__name__)

                    # log simple hyperparams
                    try:
                        params = est.get_params()
                        for k, v in params.items():
                            if isinstance(v, (str, int, float, bool)):
                                mlflow.log_param(f"hp_{k}", v)
                    except Exception:
                        pass

                    run_id = mlflow.active_run().info.run_id

                    # Train + eval
                    est.fit(X_train, y_train)
                    pred = est.predict(X_val)

                    mae, rmse, r2 = score(y_val, pred)
                    mlflow.log_metric("MAE", mae)
                    mlflow.log_metric("RMSE", rmse)
                    mlflow.log_metric("R2", r2)

                    # Save run bundle for promotion
                    bundle = {"model": est, "feature_names": feature_cols}
                    run_bundle_path = runs_dir / f"{run_id}.pkl"
                    joblib.dump(bundle, run_bundle_path)
                    mlflow.log_artifact(
                        str(run_bundle_path), artifact_path="model_bundle"
                    )

                    # Baseline stats tied to this run
                    stats = compute_baseline_stats(
                        X_train=X_train,
                        feature_cols=feature_cols,
                        run_id=run_id,
                        window=window,
                    )
                    stats_path = artifacts_dir / f"baseline_stats_{run_id}.json"
                    with open(stats_path, "w", encoding="utf-8") as f:
                        json.dump(stats, f, indent=2)
                    mlflow.log_artifact(str(stats_path), artifact_path="model")

                    all_results.append(
                        {
                            "model_key": model_key,
                            "run_id": run_id,
                            "window": window,
                            "seed": seed,
                            "MAE": mae,
                            "RMSE": rmse,
                            "R2": r2,
                        }
                    )

    # ---- Aggregate + choose best by mean RMSE (tie-break MAE) ----
    results_df = pd.DataFrame(all_results)

    agg = (
        results_df.groupby(["model_key", "window"], as_index=False)
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            R2_mean=("R2", "mean"),
            runs=("run_id", "count"),
        )
        .sort_values(["RMSE_mean", "MAE_mean"])
    )

    best_cfg = agg.iloc[0].to_dict()
    best_model = best_cfg["model_key"]
    best_window = int(best_cfg["window"])

    best_run_row = (
        results_df[
            (results_df["model_key"] == best_model)
            & (results_df["window"] == best_window)
        ]
        .sort_values(["RMSE", "MAE"])
        .iloc[0]
    )
    best_run_id = str(best_run_row["run_id"])

    # ---- Write artifacts ----
    out_dir = repo_root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_runs_path = out_dir / "benchmark_leaderboard_runs.csv"
    results_df.sort_values(["RMSE", "MAE"]).to_csv(leaderboard_runs_path, index=False)

    leaderboard_agg_path = out_dir / "benchmark_leaderboard_agg.csv"
    agg.to_csv(leaderboard_agg_path, index=False)

    summary_path = out_dir / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {"best_config": best_cfg, "best_run": best_run_row.to_dict()},
            f,
            indent=2,
        )

    # Log leaderboard artifacts to a dedicated run
    with mlflow.start_run(run_name="benchmark_summary"):
        mlflow.log_param("best_model_key", best_model)
        mlflow.log_param("best_window", best_window)
        mlflow.log_param("seeds", ",".join(map(str, seeds)))
        mlflow.log_param("windows", ",".join(map(str, windows)))
        mlflow.log_param("val_frac_by_engine", val_frac)
        mlflow.log_artifact(str(leaderboard_runs_path), artifact_path="benchmark")
        mlflow.log_artifact(str(leaderboard_agg_path), artifact_path="benchmark")
        mlflow.log_artifact(str(summary_path), artifact_path="benchmark")

    # ---- Register best as active dev (skip if DATABASE_URL missing) ----
    if os.getenv("DATABASE_URL"):
        register_model(
            run_id=best_run_id,
            artifact_path=f"/artifacts/models/runs/{best_run_id}.pkl",
            stage="dev",
            name="cmapss_fd001_rul",
            notes=f"best_mean_rmse={best_cfg['RMSE_mean']:.4f} model={best_model} window={best_window}",
        )
        print("Registered best run as active dev in model_registry.")
    else:
        print("DATABASE_URL not set; skipping dev registration.")

    print("Benchmark complete.")
    print("Best config (mean RMSE):", best_cfg)
    print("Best run:", best_run_row.to_dict())
    print("Promote via API when ready: POST /model/promote")

    cleanup_local_run_artifacts(runs_dir, artifacts_dir, keep_last=150)


if __name__ == "__main__":
    main()
