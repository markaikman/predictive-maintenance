from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from .settings import settings
import json

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(settings.database_url, pool_pre_ping=True)
    return _engine


def log_prediction(
    engine_id: str, model_version: str, prediction: float, features: dict
) -> None:
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO prediction_logs (engine_id, model_version, prediction, features)
                VALUES (:engine_id, :model_version, :prediction, CAST(:features AS jsonb))
            """
            ),
            {
                "engine_id": engine_id,
                "model_version": model_version,
                "prediction": float(prediction),
                "features": json.dumps(features),
            },
        )
