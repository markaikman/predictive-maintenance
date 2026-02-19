from __future__ import annotations

from pathlib import Path
import os
import joblib
import json
import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
from pipelines.db import get_engine


def load_cmapss_txt(path: Path) -> pd.DataFrame:
    # C-MAPSS files are space-delimited with trailing spaces; read with sep=r"\s+"
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # Drop last column if it's all NaN (common with trailing separator)
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


def write_train_to_postgres(
    raw: pd.DataFrame, table: str = "cmapss_fd001_train"
) -> None:
    # Defensive check: prevent accidental table wipe
    if raw is None or raw.empty:
        raise ValueError(
            f"Refusing to write to '{table}': dataframe is empty or None. "
            "This would replace the table with 0 rows."
        )
    # Optional sanity check for required columns
    required_cols = {"engine_id", "cycle"}
    missing = required_cols - set(raw.columns)
    if missing:
        raise ValueError(
            f"Refusing to write to '{table}': missing required columns {missing}"
        )
    eng = get_engine()
    print(f"Writing {len(raw)} rows to table '{table}'...")
    raw.to_sql(table, eng, if_exists="replace", index=False)
    print(f"Successfully wrote {len(raw)} rows to '{table}'.")


def add_rul_label(train_df: pd.DataFrame) -> pd.DataFrame:
    max_cycle = train_df.groupby("engine_id")["cycle"].max().rename("max_cycle")
    df = train_df.merge(max_cycle, on="engine_id", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def build_features(
    df: pd.DataFrame, window: int = 10
) -> tuple[pd.DataFrame, list[str]]:
    # Basic set: settings + sensors
    base_cols = [f"setting_{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]

    out = df[["engine_id", "cycle"] + base_cols].copy()

    # Rolling mean/std per engine for sensors (lightweight + strong baseline)
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


def register_model(
    run_id: str,
    artifact_path: str,
    stage: str = "dev",
    name: str = "cmapss_fd001_rul",
    notes: str | None = None,
):
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL env var is required (no hard-coded fallback).")
    eng = get_engine()

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


def train_val_split_by_engine(
    df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    engines = df["engine_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(engines)
    n_val = max(1, int(len(engines) * val_frac))
    val_engines = set(engines[:n_val])
    train = df[~df["engine_id"].isin(val_engines)].copy()
    val = df[df["engine_id"].isin(val_engines)].copy()
    return train, val


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data" / "raw"

    train_path = data_dir / "train_FD001.txt"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing {train_path}. Put NASA C-MAPSS FD001 files in data/raw/"
        )

    # MLflow config
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("cmapss_rul_fd001")

    # Load + label
    raw = load_cmapss_txt(train_path)
    write_train_to_postgres(raw)
    labeled = add_rul_label(raw)

    # Features
    feat_df, feature_cols = build_features(labeled, window=10)
    feat_df["RUL"] = labeled["RUL"].values

    # Split by engine to avoid leakage
    train_df, val_df = train_val_split_by_engine(feat_df, val_frac=0.2, seed=42)

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["RUL"].to_numpy(dtype=float)
    X_val = val_df[feature_cols].to_numpy(dtype=float)
    y_val = val_df["RUL"].to_numpy(dtype=float)

    # Baseline stats for drift monitoring (training distribution)
    baseline = {}
    X_train_df = pd.DataFrame(X_train, columns=feature_cols)

    for c in feature_cols:
        s = X_train_df[c].astype(float)
        # Robust percentiles to build bins later
        baseline[c] = {
            "p05": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
        }

    model = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1, max_depth=None
    )

    with mlflow.start_run(run_name="rf_baseline_rm10"):
        mlflow.log_param("dataset", "FD001")
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("rolling_window", 10)
        mlflow.log_param("val_frac_by_engine", 0.2)

        run_id = mlflow.active_run().info.run_id

        # 1) Train + evaluate first
        model.fit(X_train, y_train)
        pred = model.predict(X_val)

        mae = mean_absolute_error(y_val, pred)
        mse = mean_squared_error(y_val, pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_val, pred)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # 2) Now build the bundle (model exists + feature order fixed)
        bundle = {"model": model, "feature_names": feature_cols}

        artifacts_dir = repo_root / "artifacts" / "models"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # 3) Save dev bundle (optional convenience)
        dev_path = artifacts_dir / "model.pkl"
        joblib.dump(bundle, dev_path)

        # 4) Save run-specific bundle for promotion
        runs_dir = artifacts_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        run_bundle_path = runs_dir / f"{run_id}.pkl"
        joblib.dump(bundle, run_bundle_path)

        # 5) Save baseline stats next to the model artifacts
        stats_path = artifacts_dir / "baseline_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": "FD001",
                    "rolling_window": 10,
                    "feature_names": feature_cols,
                    "baseline": baseline,
                    "run_id": run_id,
                },
                f,
                indent=2,
            )

        # 6) Log artifacts to MLflow
        mlflow.log_artifact(str(run_bundle_path), artifact_path="model_bundle")
        mlflow.log_artifact(str(stats_path), artifact_path="model")

        # 7) Register in Postgres as active DEV using a container-visible path
        # IMPORTANT: store /artifacts/... not a Windows path
        register_model(
            run_id=run_id,
            artifact_path=f"/artifacts/models/runs/{run_id}.pkl",
            stage="dev",
            name="cmapss_fd001_rul",
            notes="RF baseline rm10",
        )

    print("Saved dev bundle to:", dev_path)
    print("Saved run bundle to:", run_bundle_path)
    print(f"Val MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}")
    print("MLflow Tracking URI:", mlflow_uri)


if __name__ == "__main__":
    main()
