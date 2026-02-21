from __future__ import annotations

from pathlib import Path
import os
import json
import joblib
import numpy as np
import pandas as pd

import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from sqlalchemy import create_engine, text

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


def compute_baseline_stats(X_train: np.ndarray, feature_cols: list[str]) -> dict:
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
    return baseline


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


def sample_params(rng: np.random.Generator) -> dict:
    lr = float(rng.choice([0.01, 0.02, 0.03, 0.05]))
    num_leaves = int(rng.choice([31, 63, 127, 255]))
    min_child = int(rng.choice([20, 40, 80, 120]))
    reg_lambda = float(rng.choice([1.0, 2.0, 5.0, 10.0, 20.0]))
    reg_alpha = float(rng.choice([0.0, 1e-3, 1e-2, 1e-1]))

    return {
        "n_estimators": 20000,
        "learning_rate": lr,
        "num_leaves": num_leaves,
        "min_child_samples": min_child,
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "max_depth": int(rng.choice([-1, 4, 6, 8, 10])),
        # stability
        "random_state": 42,
        "bagging_seed": 42,
        "feature_fraction_seed": 42,
        "data_random_seed": 42,
        "deterministic": True,
        "force_col_wise": True,
    }


def main():
    repo_root = Path(__file__).resolve().parents[2]
    train_path = repo_root / "data" / "raw" / "train_FD001.txt"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("cmapss_rul_fd001_tuning_multiseed")

    window = 30
    val_frac = 0.3
    seeds = [42, 123, 999, 2026, 777]
    n_trials = int(os.getenv("N_TRIALS", "1"))

    artifacts_dir = repo_root / "artifacts" / "models"
    runs_dir = artifacts_dir / "runs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    raw = load_cmapss_txt(train_path)
    labeled = add_rul_label(raw)
    feat_df, feature_cols = build_features(labeled, window=window)
    print("len labeled:", len(labeled), "len feat_df:", len(feat_df))
    assert len(feat_df) == len(labeled), "Row mismatch: features and labels not aligned"
    assert (feat_df["engine_id"].values == labeled["engine_id"].values).all()
    assert (feat_df["cycle"].values == labeled["cycle"].values).all()
    feat_df["RUL"] = labeled["RUL"].values

    rng = np.random.default_rng(2026)

    best = None  # (rmse_mean, rmse_std, r2_mean, run_id, params)

    for t in range(n_trials):
        params = sample_params(rng)
        # params = {
        #     "n_estimators": 20000,
        #     "learning_rate": 0.05,
        #     "num_leaves": 255,
        #     "min_child_samples": 20,
        #     "subsample": 0.9760451609130898,
        #     "colsample_bytree": 0.6347532221092745,
        #     "max_depth": 6,
        #     "reg_alpha": 0.0,
        #     "reg_lambda": 5.0,
        #     "random_state": 42,
        #     "bagging_seed": 42,
        #     "feature_fraction_seed": 42,
        #     "data_random_seed": 42,
        #     "deterministic": True,
        #     "force_col_wise": True,
        # }

        with mlflow.start_run(run_name=f"lgbm_ms_w{window}_t{t+1}"):
            mlflow.log_param("dataset", "FD001")
            mlflow.log_param("rolling_window", window)
            mlflow.log_param("val_frac_by_engine", val_frac)
            mlflow.log_param("seeds", ",".join(map(str, seeds)))
            for k, v in params.items():
                mlflow.log_param(f"hp_{k}", v)

            run_id = mlflow.active_run().info.run_id

            per_seed = []
            # Train/eval separately per split seed (same hyperparams)
            for split_seed in seeds:
                train_df, val_df = train_val_split_by_engine(
                    feat_df, val_frac=val_frac, seed=split_seed
                )
                X_train = train_df[feature_cols].to_numpy(dtype=float)
                y_train = train_df["RUL"].to_numpy(dtype=float)
                X_val = val_df[feature_cols].to_numpy(dtype=float)
                y_val = val_df["RUL"].to_numpy(dtype=float)

                model = LGBMRegressor(**params, n_jobs=-1)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="rmse",
                    callbacks=[
                        early_stopping(stopping_rounds=200, first_metric_only=True),
                        log_evaluation(period=0),
                    ],
                )
                best_iter = int(
                    getattr(model, "best_iteration_", params["n_estimators"])
                )
                mlflow.log_metric(f"best_iter_seed_{split_seed}", best_iter)

                pred = model.predict(X_val)
                mae, rmse, r2 = score(y_val, pred)
                per_seed.append((mae, rmse, r2))
                mlflow.log_metric(f"RMSE_seed_{split_seed}", rmse)
                mlflow.log_metric(f"MAE_seed_{split_seed}", mae)
                mlflow.log_metric(f"R2_seed_{split_seed}", r2)

            maes = np.array([x[0] for x in per_seed])
            rmses = np.array([x[1] for x in per_seed])
            r2s = np.array([x[2] for x in per_seed])

            rmse_mean = float(rmses.mean())
            rmse_std = float(rmses.std(ddof=0))
            mae_mean = float(maes.mean())
            r2_mean = float(r2s.mean())

            mlflow.log_metric("RMSE_mean", rmse_mean)
            mlflow.log_metric("RMSE_std", rmse_std)
            mlflow.log_metric("MAE_mean", mae_mean)
            mlflow.log_metric("R2_mean", r2_mean)

            # Fit one final model on the "default" seed split to create a bundle artifact
            # (we promote a specific model artifact; the selection is based on mean metrics)
            train_df, val_df = train_val_split_by_engine(
                feat_df, val_frac=val_frac, seed=seeds[0]
            )
            X_train = train_df[feature_cols].to_numpy(dtype=float)
            y_train = train_df["RUL"].to_numpy(dtype=float)
            X_val = val_df[feature_cols].to_numpy(dtype=float)
            y_val = val_df["RUL"].to_numpy(dtype=float)

            final_model = LGBMRegressor(**params, n_jobs=-1)
            final_model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=[
                    early_stopping(stopping_rounds=200, first_metric_only=True),
                    log_evaluation(period=0),
                ],
            )
            mlflow.log_metric(
                "best_iteration_final",
                int(getattr(final_model, "best_iteration_", params["n_estimators"])),
            )

            bundle = {"model": final_model, "feature_names": feature_cols}
            run_bundle_path = runs_dir / f"{run_id}.pkl"
            joblib.dump(bundle, run_bundle_path)
            mlflow.log_artifact(str(run_bundle_path), artifact_path="model_bundle")

            # Save baseline stats for this run
            baseline_stats = compute_baseline_stats(X_train, feature_cols)
            payload = {
                "dataset": "FD001",
                "rolling_window": window,
                "feature_names": feature_cols,
                "baseline": baseline_stats,
                "run_id": run_id,
            }
            stats_path = artifacts_dir / f"baseline_stats_{run_id}.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            mlflow.log_artifact(str(stats_path), artifact_path="model")

            # Track best: mean RMSE primary, std RMSE secondary
            if (
                best is None
                or (rmse_mean < best[0])
                or (rmse_mean == best[0] and rmse_std < best[1])
                or (rmse_mean == best[0] and rmse_std == best[1] and r2_mean > best[2])
            ):
                best = (rmse_mean, rmse_std, r2_mean, run_id, params)

    if not best:
        raise RuntimeError("No trials completed")

    best_rmse_mean, best_rmse_std, best_r2_mean, best_run_id, best_params = best
    print(
        f"Best trial: run_id={best_run_id} RMSE_mean={best_rmse_mean:.4f} RMSE_std={best_rmse_std:.4f} R2_mean={best_r2_mean:.4f}"
    )

    if os.getenv("DATABASE_URL"):
        register_model(
            run_id=best_run_id,
            artifact_path=f"/artifacts/models/runs/{best_run_id}.pkl",
            stage="dev",
            name="cmapss_fd001_rul",
            notes=f"lgbm_multiseed rmse_mean={best_rmse_mean:.4f} std={best_rmse_std:.4f} window={window}",
        )
        print("Registered best multiseed tuned model as active dev.")
    else:
        print("DATABASE_URL not set; skipping dev registration.")

    print("Promote via API when ready: POST /model/promote")


if __name__ == "__main__":
    main()
