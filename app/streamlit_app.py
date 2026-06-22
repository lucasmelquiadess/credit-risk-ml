from __future__ import annotations

import json
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from src.models.evaluate_model import predict_positive_probability
from src.utils.config import DEFAULT_MODEL_PATH

st.set_page_config(page_title="Credit Risk Prediction", layout="centered")


@st.cache_resource
def load_artifact() -> dict[str, Any] | None:
    if not DEFAULT_MODEL_PATH.exists():
        return None
    return joblib.load(DEFAULT_MODEL_PATH)


sample_record = {
    "age": 35,
    "income": 72000,
    "loan_amount": 18000,
    "loan_term_months": 36,
    "interest_rate": 0.12,
    "employment_years": 6,
    "credit_history_years": 8,
    "existing_debt": 9000,
    "missed_payments_2y": 0,
    "has_mortgage": 1,
    "loan_purpose": "car",
}

st.title("Credit Risk Prediction")
st.caption("Demonstração local para prever probabilidade de inadimplência.")

artifact = load_artifact()

if artifact is None:
    st.warning(
        "Nenhum modelo treinado foi encontrado. Rode o script de treino antes de usar "
        "a interface."
    )
    st.code(
        "python -m src.data.make_dataset --use-sample\n"
        "python -m src.models.train_model "
        "--input-path data/processed/credit_risk_dataset.csv",
        language="powershell",
    )
else:
    metadata = artifact.get("metadata", {})
    st.info(f"Modelo carregado: {metadata.get('best_model', 'modelo sem metadados')}")

input_text = st.text_area(
    "Registro para predição em JSON",
    value=json.dumps(sample_record, indent=2),
    height=260,
)
threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if st.button("Prever risco", type="primary", disabled=artifact is None):
    try:
        record = json.loads(input_text)
        records = record if isinstance(record, list) else [record]
        X = pd.DataFrame(records)
        probabilities = predict_positive_probability(artifact["pipeline"], X)
        predictions = (probabilities >= threshold).astype(int)

        output = pd.DataFrame(
            {
                "predicted_default_probability": probabilities.round(6),
                "predicted_default": predictions,
            }
        )
        st.dataframe(output, use_container_width=True)
    except Exception as exc:
        st.error(f"Não foi possível gerar a predição: {exc}")
