from pathlib import Path
import joblib
import numpy as np
from typing import Any


class DummyModel:
    """Fallback model so the API runs before training on real data."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sum(X, axis=1) * 0.01


def load_model_bundle(model_path: str) -> dict[str, Any]:
    p = Path(model_path)
    if p.exists():
        obj = joblib.load(p)
        # Expect {"model": estimator, "feature_names": [...]}
        if isinstance(obj, dict) and "model" in obj and "feature_names" in obj:
            return obj
        # Back-compat: raw estimator
        return {"model": obj, "feature_names": None}
    return {"model": DummyModel(), "feature_names": None}
