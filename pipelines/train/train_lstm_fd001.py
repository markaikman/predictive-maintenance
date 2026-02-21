from __future__ import annotations

from pathlib import Path
import os
import json
import time
import numpy as np
import pandas as pd

import mlflow
import torch
import torch.nn as nn

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipelines.common.cmapss_fd001 import (
    load_cmapss_txt,
    add_rul_label,
    train_val_split_by_engine,
)

from pipelines.torch.dataset_fd001 import SeqConfig, build_dataloaders
from pipelines.torch.models import LSTMRegressor


def score(y_true: np.ndarray, y_pred: np.ndarray):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return mae, rmse, r2


def build_sequence_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    For LSTM we start with per-cycle raw signals + cycle_norm.
    Uses settings + sensors + cycle_norm.
    """
    base_cols = [f"setting_{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
    out = df[["engine_id", "cycle"] + base_cols].copy()
    out["cycle_norm"] = out["cycle"] / out.groupby("engine_id")["cycle"].transform(
        "max"
    )
    feature_cols = [c for c in out.columns if c not in ("engine_id", "cycle")]
    return out, feature_cols


def train_one_seed(
    feat_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    seq_len: int,
    val_frac: float,
    split_seed: int,
    device: str,
    max_epochs: int,
    patience: int,
    lr: float,
    batch_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
):
    train_df, val_df = train_val_split_by_engine(
        feat_df, val_frac=val_frac, seed=split_seed
    )

    cfg = SeqConfig(seq_len=seq_len, batch_size=batch_size, num_workers=0)
    train_loader, val_loader, scaler_meta = build_dataloaders(
        train_df, val_df, feature_cols, cfg
    )

    model = LSTMRegressor(
        n_features=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    best_epoch = -1
    bad = 0

    def eval_rmse() -> tuple[float, np.ndarray, np.ndarray]:
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                ys.append(yb.detach().cpu().numpy())
                ps.append(pred.detach().cpu().numpy())
        y_true = np.concatenate(ys) if ys else np.array([])
        y_pred = np.concatenate(ps) if ps else np.array([])
        rmse = (
            float(np.sqrt(mean_squared_error(y_true, y_pred)))
            if len(y_true)
            else float("inf")
        )
        return rmse, y_true, y_pred

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        val_rmse, _, _ = eval_rmse()

        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    final_rmse, y_true, y_pred = eval_rmse()
    mae, rmse, r2 = score(y_true, y_pred)

    info = {
        "split_seed": split_seed,
        "best_epoch": best_epoch,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }
    return model, scaler_meta, info


def main():
    repo_root = Path(__file__).resolve().parents[2]
    train_path = repo_root / "data" / "raw" / "train_FD001.txt"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("cmapss_rul_fd001_pytorch_lstm")

    # Config
    seq_len = int(os.getenv("SEQ_LEN", "30"))
    val_frac = float(os.getenv("VAL_FRAC", "0.3"))
    seeds = [42, 123, 999, 2026, 777]

    max_epochs = int(os.getenv("EPOCHS", "25"))
    patience = int(os.getenv("PATIENCE", "4"))
    lr = float(os.getenv("LR", "0.001"))
    batch_size = int(os.getenv("BATCH_SIZE", "256"))
    hidden_size = int(os.getenv("HIDDEN_SIZE", "64"))
    num_layers = int(os.getenv("NUM_LAYERS", "2"))
    dropout = float(os.getenv("DROPOUT", "0.2"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw = load_cmapss_txt(train_path)
    labeled = add_rul_label(raw)

    feat_df, feature_cols = build_sequence_features(labeled)
    # align target
    feat_df["RUL"] = labeled["RUL"].values

    artifacts_dir = repo_root / "artifacts" / "models"
    runs_dir = artifacts_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"lstm_seq{seq_len}_vf{val_frac}"):
        mlflow.log_param("dataset", "FD001")
        mlflow.log_param("model_family", "pytorch_lstm")
        mlflow.log_param("seq_len", seq_len)
        mlflow.log_param("val_frac_by_engine", val_frac)
        mlflow.log_param("seeds", ",".join(map(str, seeds)))
        mlflow.log_param("device", device)

        mlflow.log_param("epochs", max_epochs)
        mlflow.log_param("patience", patience)
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("dropout", dropout)

        run_id = mlflow.active_run().info.run_id

        per_seed = []
        best_seed_model = None
        best_seed_meta = None
        best_seed_rmse = float("inf")
        best_seed = None

        t0 = time.time()

        for s in seeds:
            model, meta, info = train_one_seed(
                feat_df,
                feature_cols,
                seq_len=seq_len,
                val_frac=val_frac,
                split_seed=s,
                device=device,
                max_epochs=max_epochs,
                patience=patience,
                lr=lr,
                batch_size=batch_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )

            per_seed.append(info)
            mlflow.log_metric(f"RMSE_seed_{s}", info["RMSE"])
            mlflow.log_metric(f"MAE_seed_{s}", info["MAE"])
            mlflow.log_metric(f"R2_seed_{s}", info["R2"])
            mlflow.log_metric(f"best_epoch_seed_{s}", info["best_epoch"])

            if info["RMSE"] < best_seed_rmse:
                best_seed_rmse = info["RMSE"]
                best_seed_model = model
                best_seed_meta = meta
                best_seed = s

        rmses = np.array([x["RMSE"] for x in per_seed], dtype=float)
        maes = np.array([x["MAE"] for x in per_seed], dtype=float)
        r2s = np.array([x["R2"] for x in per_seed], dtype=float)

        mlflow.log_metric("RMSE_mean", float(rmses.mean()))
        mlflow.log_metric("RMSE_std", float(rmses.std(ddof=0)))
        mlflow.log_metric("MAE_mean", float(maes.mean()))
        mlflow.log_metric("R2_mean", float(r2s.mean()))
        mlflow.log_metric("train_seconds", float(time.time() - t0))
        mlflow.log_param(
            "best_seed_saved", int(best_seed) if best_seed is not None else -1
        )

        # Save a bundle: model weights + metadata needed to reproduce preprocessing
        out_path = runs_dir / f"{run_id}.pt"
        bundle = {
            "run_id": run_id,
            "model_state_dict": (
                best_seed_model.state_dict() if best_seed_model else None
            ),
            "model_config": {
                "n_features": len(feature_cols),
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
            },
            "preprocess": best_seed_meta,
            "metrics": {
                "per_seed": per_seed,
                "RMSE_mean": float(rmses.mean()),
                "RMSE_std": float(rmses.std(ddof=0)),
                "MAE_mean": float(maes.mean()),
                "R2_mean": float(r2s.mean()),
            },
        }
        torch.save(bundle, out_path)

        mlflow.log_artifact(str(out_path), artifact_path="model_bundle")

        # Also log a small JSON summary (easy to view without torch)
        summary_path = artifacts_dir / f"lstm_summary_{run_id}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(bundle["metrics"], f, indent=2)
        mlflow.log_artifact(str(summary_path), artifact_path="model")

        print(f"Saved LSTM bundle: {out_path}")
        print("Per-seed RMSE:", {x["split_seed"]: x["RMSE"] for x in per_seed})
        print(f"RMSE_mean={rmses.mean():.4f} RMSE_std={rmses.std(ddof=0):.4f}")


if __name__ == "__main__":
    main()
