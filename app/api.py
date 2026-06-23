from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.features.build_features import build_features
from src.models.evaluate_model import predict_positive_probability
from src.utils.config import DEFAULT_MODEL_PATH

OBSERVATION = (
    "Experimental portfolio prediction. This API is for learning and demonstration "
    "only, not for real credit decisions."
)

app = FastAPI(
    title="Credit Risk Prediction API",
    version="0.1.0",
    description="Experimental API for binary credit default risk prediction.",
)


class CreditApplicant(BaseModel):
    model_config = ConfigDict(extra="allow")

    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Applicant total income.")
    AMT_CREDIT: float = Field(..., gt=0, description="Requested credit amount.")
    AMT_ANNUITY: float = Field(..., gt=0, description="Loan annuity amount.")
    DAYS_BIRTH: int = Field(
        ...,
        lt=0,
        description="Applicant age in days before application. Home Credit stores this as a negative number.",
    )
    DAYS_EMPLOYED: int = Field(
        ...,
        description="Days employed before application. Home Credit usually stores this as a negative number.",
    )

    NAME_CONTRACT_TYPE: str = "Cash loans"
    CODE_GENDER: str = "F"
    FLAG_OWN_CAR: str = "N"
    FLAG_OWN_REALTY: str = "Y"
    CNT_CHILDREN: int = Field(default=0, ge=0)
    NAME_INCOME_TYPE: str = "Working"
    NAME_EDUCATION_TYPE: str = "Secondary / secondary special"
    NAME_FAMILY_STATUS: str = "Married"
    NAME_HOUSING_TYPE: str = "House / apartment"
    AMT_GOODS_PRICE: float | None = Field(default=None, gt=0)


class PredictionResponse(BaseModel):
    default_probability: float
    risk_level: str
    observation: str
    model_name: str | None = None


@lru_cache(maxsize=1)
def load_artifact() -> dict[str, Any]:
    if not DEFAULT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {DEFAULT_MODEL_PATH}. "
            "Train a model with python src/models/train_model.py first."
        )
    return joblib.load(DEFAULT_MODEL_PATH)


def get_preprocessor_columns(artifact: dict[str, Any]) -> tuple[list[str], list[str]]:
    preprocessor = artifact["pipeline"].named_steps["preprocess"]
    columns_by_transformer = {
        name: list(columns)
        for name, _, columns in preprocessor.transformers_
        if name != "remainder"
    }
    return (
        columns_by_transformer.get("numeric", []),
        columns_by_transformer.get("categorical", []),
    )


def build_model_input(payload: CreditApplicant, artifact: dict[str, Any]) -> pd.DataFrame:
    numeric_columns, categorical_columns = get_preprocessor_columns(artifact)
    expected_columns = artifact.get("metadata", {}).get("feature_columns", [])

    record: dict[str, Any] = {}
    record.update({column: np.nan for column in numeric_columns})
    record.update({column: "Unknown" for column in categorical_columns})

    payload_values = payload.model_dump(exclude_none=True)
    if "AMT_GOODS_PRICE" not in payload_values:
        payload_values["AMT_GOODS_PRICE"] = payload.AMT_CREDIT

    record.update(payload_values)

    input_df = pd.DataFrame([record])
    input_df = build_features(input_df)

    if expected_columns:
        missing_columns = sorted(set(expected_columns) - set(input_df.columns))
        if missing_columns:
            raise ValueError(f"Missing model input columns: {missing_columns}")
        input_df = input_df[expected_columns]

    return input_df


def classify_risk(default_probability: float) -> str:
    if default_probability < 0.30:
        return "low"
    if default_probability < 0.60:
        return "medium"
    return "high"


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "Credit Risk Prediction API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CreditApplicant) -> PredictionResponse:
    try:
        artifact = load_artifact()
        input_df = build_model_input(payload, artifact)
        pipeline = artifact["pipeline"]
        probability = float(predict_positive_probability(pipeline, input_df)[0])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictionResponse(
        default_probability=round(probability, 6),
        risk_level=classify_risk(probability),
        observation=OBSERVATION,
        model_name=artifact.get("metadata", {}).get("best_model"),
    )
