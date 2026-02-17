from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sqlalchemy import text

from .db import get_engine
from .settings import settings


def load_baseline_stats() -> dict:
    stats_path = Path(settings.model_path).with_name("baseline_stats.json")
    if not stats_path.exists():
        return {}
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _psi(expected: np.ndarray, actual: np.ndarray, bins: np.ndarray) -> float:
    """
    Population Stability Index.
    expected: training distribution sample
    actual: current distribution sample
    bins: bin edges
    """
    eps = 1e-6
    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    exp_perc = exp_counts / max(exp_counts.sum(), 1)
    act_perc = act_counts / max(act_counts.sum(), 1)

    exp_perc = np.clip(exp_perc, eps, 1.0)
    act_perc = np.clip(act_perc, eps, 1.0)

    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))


def get_recent_prediction_logs(limit: int = 1000) -> pd.DataFrame:
    eng = get_engine()
    q = text(
        """
        SELECT created_at, engine_id, model_version, prediction, features
        FROM prediction_logs
        ORDER BY created_at DESC
        LIMIT :limit
    """
    )
    df = pd.read_sql(q, eng, params={"limit": limit})
    return df


def compute_feature_drift_psi(limit: int = 1000) -> dict:
    stats = load_baseline_stats()
    if not stats:
        return {
            "ok": False,
            "error": "baseline_stats.json not found. Re-run training to generate it.",
        }

    df = get_recent_prediction_logs(limit=limit)
    if df.empty:
        return {"ok": False, "error": "No prediction logs found."}

    # Expand JSON features into columns
    feat_df = pd.json_normalize(df["features"])
    feat_df = feat_df.apply(pd.to_numeric, errors="coerce")

    feature_names = stats.get("feature_names", [])
    baseline = stats.get("baseline", {})

    drift = []
    for f in feature_names:
        if f not in feat_df.columns or f not in baseline:
            continue

        actual = feat_df[f].dropna().to_numpy(dtype=float)
        if actual.size < 50:
            continue

        # Build bins from training percentiles
        p05 = baseline[f]["p05"]
        p25 = baseline[f]["p25"]
        p50 = baseline[f]["p50"]
        p75 = baseline[f]["p75"]
        p95 = baseline[f]["p95"]

        # Make sure bins are strictly increasing
        edges = np.array([p05, p25, p50, p75, p95], dtype=float)
        edges = np.unique(edges)
        if edges.size < 3:
            continue

        # Add -inf/+inf edges
        bins = np.concatenate(([-np.inf], edges, [np.inf]))

        # Approximate expected distribution by sampling from N(mean,std) (lightweight)
        mu = baseline[f]["mean"]
        sd = max(baseline[f]["std"], 1e-6)
        expected = np.random.default_rng(42).normal(mu, sd, size=5000)

        psi = _psi(expected, actual, bins=bins)
        drift.append({"feature": f, "psi": psi, "n": int(actual.size)})

    drift.sort(key=lambda x: x["psi"], reverse=True)
    return {
        "ok": True,
        "window_n": int(min(limit, len(df))),
        "top": drift[:25],
        "all_count": len(drift),
        "guidance": {
            "psi_low": "PSI < 0.10: little/no drift",
            "psi_med": "0.10–0.25: moderate drift (investigate)",
            "psi_high": "PSI > 0.25: high drift (potential issue)",
        },
    }
