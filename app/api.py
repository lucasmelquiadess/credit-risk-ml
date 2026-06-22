from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.evaluate_model import predict_positive_probability
from src.utils.config import DEFAULT_MODEL_PATH

app = FastAPI(title="Credit Risk Prediction API", version="0.1.0")


class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


@lru_cache(maxsize=1)
def load_artifact() -> dict[str, Any]:
    if not DEFAULT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {DEFAULT_MODEL_PATH}. Train a model first."
        )
    return joblib.load(DEFAULT_MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, Any]:
    try:
        artifact = load_artifact()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    pipeline = artifact["pipeline"]
    X = pd.DataFrame(payload.records)
    probabilities = predict_positive_probability(pipeline, X)
    predictions = (probabilities >= payload.threshold).astype(int)

    return {
        "model_metadata": artifact.get("metadata", {}),
        "predictions": [
            {
                "predicted_default": int(prediction),
                "predicted_default_probability": round(float(probability), 6),
            }
            for prediction, probability in zip(predictions, probabilities, strict=True)
        ],
    }
