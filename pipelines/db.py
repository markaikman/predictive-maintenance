import os
from sqlalchemy import create_engine, Engine


def get_engine() -> Engine:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL env var is required.")
    return create_engine(db_url, pool_pre_ping=True)
