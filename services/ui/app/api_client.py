import os
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def predict(engine_id: str, features: dict) -> dict:
    r = requests.post(
        f"{API_BASE_URL}/predict",
        json={"engine_id": engine_id, "features": features},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()
