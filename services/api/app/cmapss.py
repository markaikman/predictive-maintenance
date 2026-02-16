from __future__ import annotations

from typing import Sequence
import pandas as pd
from sqlalchemy import text

from .db import get_engine

BASE_COLS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)


def list_engines(table: str = "cmapss_fd001_train") -> list[int]:
    eng = get_engine()
    with eng.begin() as conn:
        rows = conn.execute(
            text(f"SELECT DISTINCT engine_id FROM {table} ORDER BY engine_id")
        ).fetchall()
    return [int(r[0]) for r in rows]


def get_engine_series(
    engine_id: int, cols: Sequence[str], table: str = "cmapss_fd001_train"
) -> pd.DataFrame:
    cols = ["cycle"] + [c for c in cols if c != "cycle"]
    # Basic validation: only allow known columns
    allowed = set(BASE_COLS)
    for c in cols:
        if c not in allowed:
            raise ValueError(f"Invalid column: {c}")

    col_sql = ", ".join(cols)
    q = text(f"SELECT {col_sql} FROM {table} WHERE engine_id = :eid ORDER BY cycle")
    eng = get_engine()
    df = pd.read_sql(q, eng, params={"eid": engine_id})
    return df
