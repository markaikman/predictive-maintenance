from pydantic import BaseModel, Field
from typing import Dict


class PredictRequest(BaseModel):
    engine_id: str = Field(..., examples=["Engine_12"])
    features: Dict[str, float]


class PredictResponse(BaseModel):
    engine_id: str
    prediction: float
    model_version: str
