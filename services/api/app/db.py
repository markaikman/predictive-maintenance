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
    engine = get_engine()
    with engine.begin() as conn:
        row = (
            conn.execute(
                text(
                    """
                SELECT run_id, artifact_path, notes, created_at
                FROM model_registry
                WHERE name = :name
                  AND stage = :stage
                  AND is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 1
            """
                ),
                {"name": name, "stage": stage},
            )
            .mappings()
            .first()
        )

    if not row:
        return None

    d = dict(row)
    if d.get("created_at") is not None:
        d["created_at"] = d["created_at"].isoformat()
    return d


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
    """
    Promote the currently-active dev row to prod.
    Ensures:
      - exactly one active prod row
      - dev row that was promoted becomes inactive
    Returns dict with promoted run_id + artifact_path.
    """
    eng = get_engine()
    with eng.begin() as conn:
        # 1) fetch active dev
        dev = (
            conn.execute(
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
            )
            .mappings()
            .first()
        )

        if not dev:
            raise ValueError(f"No active dev model found for name={name}")

        run_id = dev["run_id"]
        artifact_path = dev["artifact_path"]
        dev_notes = dev.get("notes")

        # 2) deactivate current active prod
        conn.execute(
            text(
                """
                UPDATE model_registry
                SET is_active = FALSE
                WHERE name = :name AND stage = 'prod' AND is_active = TRUE
            """
            ),
            {"name": name},
        )

        # 3) insert new prod row (active)
        combined_notes = notes if notes is not None else dev_notes
        conn.execute(
            text(
                """
                INSERT INTO model_registry (name, stage, run_id, artifact_path, notes, is_active)
                VALUES (:name, 'prod', :run_id, :artifact_path, :notes, TRUE)
            """
            ),
            {
                "name": name,
                "run_id": run_id,
                "artifact_path": artifact_path,
                "notes": combined_notes,
            },
        )

        # 4) deactivate the dev row that was promoted
        conn.execute(
            text(
                """
                UPDATE model_registry
                SET is_active = FALSE
                WHERE id = :id
            """
            ),
            {"id": dev["id"]},
        )

    return {"run_id": run_id, "artifact_path": artifact_path, "notes": combined_notes}
