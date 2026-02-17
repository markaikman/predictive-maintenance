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


def get_active_model(name: str, stage: str) -> dict | None:
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(
            text(
                """
                SELECT id, created_at, name, stage, run_id, artifact_path, notes
                FROM model_registry
                WHERE name = :name AND stage = :stage AND is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 1
            """
            ),
            {"name": name, "stage": stage},
        ).fetchone()
    if not row:
        return None
    return dict(row._mapping)


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


def promote_dev_to_prod(name: str, notes: str | None = None) -> dict:
    eng = get_engine()
    with eng.begin() as conn:
        dev = conn.execute(
            text(
                """
                SELECT id, run_id, artifact_path, notes
                FROM model_registry
                WHERE name = :name AND stage = 'dev' AND is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 1
            """
            ),
            {"name": name},
        ).fetchone()
        if not dev:
            raise RuntimeError("No active dev model to promote")

        # deactivate existing prod
        conn.execute(
            text(
                "UPDATE model_registry SET is_active = FALSE WHERE name = :name AND stage = 'prod' AND is_active = TRUE"
            ),
            {"name": name},
        )

        # insert new prod row
        conn.execute(
            text(
                """
                INSERT INTO model_registry (name, stage, run_id, artifact_path, notes, is_active)
                VALUES (:name, 'prod', :run_id, :artifact_path, :notes, TRUE)
            """
            ),
            {
                "name": name,
                "run_id": dev._mapping["run_id"],
                "artifact_path": dev._mapping["artifact_path"],
                "notes": notes or dev._mapping.get("notes"),
            },
        )
        return {
            "run_id": dev._mapping["run_id"],
            "artifact_path": dev._mapping["artifact_path"],
        }
