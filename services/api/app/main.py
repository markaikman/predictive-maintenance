from fastapi import FastAPI, HTTPException, Query, Header
from pathlib import Path
import shutil
import numpy as np
from typing import List
from sqlalchemy import text
from .schemas import PredictRequest, PredictResponse
from .settings import settings
from .db import log_prediction, get_engine, get_active_model, promote_dev_to_prod
from .cmapss import list_engines, get_engine_series
from .features import build_features_for_engine
from .monitoring import get_recent_prediction_logs, compute_feature_drift_psi
from .model_loader import load_bundle_from_path

app = FastAPI(title="Predictive Maintenance API")

bundle = load_bundle_from_path(settings.production_model_path)
model = bundle["model"]
feature_names = bundle["feature_names"]
current_model_source = (
    "production.pkl" if Path(settings.production_model_path).exists() else "fallback"
)


def _require_admin(x_api_key: str | None):
    if not settings.admin_api_key:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY not set")
    if not x_api_key or x_api_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health():
    return {"status": "ok", "model_path": settings.model_path}


@app.get("/engines")
def engines():
    try:
        return {"engines": list_engines()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/predictions")
def monitoring_predictions(limit: int = 1000):
    df = get_recent_prediction_logs(limit=limit)
    return {"rows": df.to_dict(orient="records")}


@app.get("/monitoring/feature-drift")
def monitoring_feature_drift(limit: int = 1000):
    return compute_feature_drift_psi(limit=limit)


@app.get("/engines/{engine_id}/series")
def engine_series(engine_id: int, cols: List[str] = Query(default=["s1", "s2", "s3"])):
    try:
        df = get_engine_series(engine_id, cols=cols)
        # Return records for Streamlit
        return {"engine_id": engine_id, "rows": df.to_dict(orient="records")}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
def model_info():
    prod = get_active_model("cmapss_fd001_rul", "prod")
    dev = get_active_model("cmapss_fd001_rul", "dev")
    return {
        "loaded_from": current_model_source,
        "feature_names_count": 0 if not feature_names else len(feature_names),
        "feature_names_sample": [] if not feature_names else feature_names[:10],
        "registry": {"dev": dev, "prod": prod},
    }


@app.post("/engines/{engine_id}/predict-latest")
def predict_latest(engine_id: int, window: int = 10):
    try:
        # Pull full engine history needed for feature building
        eng = get_engine()
        q = text(
            """
            SELECT engine_id, cycle,
                   setting_1, setting_2, setting_3,
                   s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11,
                   s12, s13, s14, s15, s16, s17, s18, s19, s20, s21
            FROM cmapss_fd001_train
            WHERE engine_id = :eid
            ORDER BY cycle
        """
        )
        import pandas as pd

        df = pd.read_sql(q, eng, params={"eid": engine_id})
        if df.empty:
            raise HTTPException(
                status_code=404, detail=f"engine_id {engine_id} not found"
            )

        feat_df, feat_cols = build_features_for_engine(df, window=window)

        # Get the latest cycle feature vector
        latest = feat_df.iloc[-1]
        features = {c: float(latest[c]) for c in feat_cols}

        # Use model bundle feature order if present
        if feature_names:
            missing = [f for f in feature_names if f not in features]
            if missing:
                raise HTTPException(
                    status_code=500,
                    detail=f"Missing features for model: {missing[:5]} ...",
                )
            x = np.array([[features[f] for f in feature_names]], dtype=float)
        else:
            x = np.array([[features[c] for c in feat_cols]], dtype=float)

        y = float(model.predict(x)[0])

        log_prediction(
            engine_id=str(engine_id),
            model_version=settings.model_version,
            prediction=y,
            features=features,
        )

        return {
            "engine_id": engine_id,
            "cycle": int(df["cycle"].iloc[-1]),
            "prediction_rul": y,
            "window": window,
            "model_version": settings.model_version,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if feature_names:
        missing = [f for f in feature_names if f not in req.features]
        if missing:
            # FastAPI will return 422 if we raise properly; keep simple for now
            return PredictResponse(
                engine_id=req.engine_id,
                prediction=float("nan"),
                model_version=settings.model_version,
            )
        x = np.array([[req.features[f] for f in feature_names]], dtype=float)
    else:
        keys = sorted(req.features.keys())
        x = np.array([[req.features[k] for k in keys]], dtype=float)

    y = float(model.predict(x)[0])

    log_prediction(
        engine_id=req.engine_id,
        model_version=settings.model_version,
        prediction=y,
        features=req.features,
    )

    return PredictResponse(
        engine_id=req.engine_id, prediction=y, model_version=settings.model_version
    )


@app.post("/model/reload")
def model_reload(x_api_key: str | None = Header(default=None)):
    _require_admin(x_api_key)

    global model, feature_names, current_model_source
    # Prefer production model; fallback to dev path
    if Path(settings.production_model_path).exists():
        bundle = load_bundle_from_path(settings.production_model_path)
        current_model_source = "production.pkl"
    else:
        bundle = load_bundle_from_path(settings.model_path)
        current_model_source = "model.pkl"

    model = bundle["model"]
    feature_names = bundle["feature_names"]
    return {"ok": True, "loaded_from": current_model_source}


@app.post("/model/promote")
def model_promote(
    x_api_key: str | None = Header(default=None), notes: str | None = None
):
    _require_admin(x_api_key)

    # 1) promote dev → prod in DB
    promoted = promote_dev_to_prod("cmapss_fd001_rul", notes=notes)

    # 2) copy bundle to production.pkl
    src = Path(promoted["artifact_path"])
    dst = Path(settings.production_model_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise HTTPException(
            status_code=500, detail=f"Promoted artifact_path missing: {src}"
        )
    shutil.copyfile(src, dst)

    # 3) reload model in memory
    bundle = load_bundle_from_path(str(dst))
    global model, feature_names, current_model_source
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    current_model_source = "production.pkl"

    return {
        "ok": True,
        "promoted_run_id": promoted["run_id"],
        "production_path": str(dst),
    }


@app.get("/health/db")
def health_db():
    # Simple write/read test
    from sqlalchemy import text
    from .db import get_engine

    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text("SELECT 1"))
    return {"db": "ok"}
