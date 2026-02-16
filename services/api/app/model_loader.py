from pathlib import Path
import joblib
import numpy as np


class DummyModel:
    """Fallback model so the API runs before you train a real one."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sum(X, axis=1) * 0.01


def load_model(model_path: str):
    p = Path(model_path)
    if p.exists():
        return joblib.load(p)
    return DummyModel()
