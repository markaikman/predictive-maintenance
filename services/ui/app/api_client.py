import os
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def list_engines() -> list[int]:
    r = requests.get(f"{API_BASE_URL}/engines", timeout=10)
    r.raise_for_status()
    return r.json()["engines"]


def get_series(engine_id: int, cols: list[str]) -> list[dict]:
    params = [("cols", c) for c in cols]
    r = requests.get(
        f"{API_BASE_URL}/engines/{engine_id}/series", params=params, timeout=20
    )
    r.raise_for_status()
    return r.json()["rows"]


def predict_latest(engine_id: int, window: int = 10) -> dict:
    r = requests.post(
        f"{API_BASE_URL}/engines/{engine_id}/predict-latest",
        params={"window": window},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()
