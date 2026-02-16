from fastapi import FastAPI
import numpy as np

from .schemas import PredictRequest, PredictResponse
from .settings import settings
from .model_loader import load_model
from .db import log_prediction

app = FastAPI(title="Predictive Maintenance API")

model = load_model(settings.model_path)


@app.get("/health")
def health():
    return {"status": "ok", "model_path": settings.model_path}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Stable feature ordering
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


@app.get("/health/db")
def health_db():
    # Simple write/read test
    from sqlalchemy import text
    from .db import get_engine

    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text("SELECT 1"))
    return {"db": "ok"}
