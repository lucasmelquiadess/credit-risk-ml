from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.api import (  # noqa: E402
    OBSERVATION,
    CreditApplicant,
    build_model_input,
    classify_risk,
    load_artifact,
)
from src.models.evaluate_model import predict_positive_probability  # noqa: E402
from src.utils.config import DEFAULT_METRICS_PATH, DEFAULT_MODEL_PATH  # noqa: E402

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="",
    layout="wide",
)


@st.cache_resource
def get_model_artifact() -> dict[str, Any] | None:
    try:
        return load_artifact()
    except FileNotFoundError:
        return None


@st.cache_data
def load_metrics() -> pd.DataFrame | None:
    if not DEFAULT_METRICS_PATH.exists():
        return None
    return pd.read_csv(DEFAULT_METRICS_PATH)


def format_probability(value: float) -> str:
    return f"{value:.2%}"


def build_applicant_from_form() -> CreditApplicant:
    with st.form("credit_application_form"):
        st.subheader("Simulação de solicitação")

        col_left, col_right = st.columns(2)
        with col_left:
            income = st.number_input(
                "Renda total",
                min_value=1_000.0,
                max_value=5_000_000.0,
                value=180_000.0,
                step=5_000.0,
            )
            credit = st.number_input(
                "Valor do crédito",
                min_value=1_000.0,
                max_value=5_000_000.0,
                value=600_000.0,
                step=10_000.0,
            )
            annuity = st.number_input(
                "Valor da anuidade",
                min_value=100.0,
                max_value=1_000_000.0,
                value=28_000.0,
                step=1_000.0,
            )
            goods_price = st.number_input(
                "Valor do bem financiado",
                min_value=1_000.0,
                max_value=5_000_000.0,
                value=600_000.0,
                step=10_000.0,
            )

        with col_right:
            age_years = st.number_input(
                "Idade em anos",
                min_value=18,
                max_value=90,
                value=38,
                step=1,
            )
            employed_years = st.number_input(
                "Anos empregado",
                min_value=0.0,
                max_value=60.0,
                value=7.0,
                step=0.5,
            )
            contract_type = st.selectbox(
                "Tipo de contrato",
                options=["Cash loans", "Revolving loans"],
                index=0,
            )
            owns_car = st.selectbox("Possui carro?", options=["N", "Y"], index=0)
            owns_realty = st.selectbox("Possui imóvel?", options=["Y", "N"], index=0)
            children = st.number_input(
                "Quantidade de filhos",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
            )

        submitted = st.form_submit_button("Calcular risco", type="primary")

    applicant = CreditApplicant(
        AMT_INCOME_TOTAL=income,
        AMT_CREDIT=credit,
        AMT_ANNUITY=annuity,
        AMT_GOODS_PRICE=goods_price,
        DAYS_BIRTH=-int(age_years * 365.25),
        DAYS_EMPLOYED=-int(employed_years * 365.25),
        NAME_CONTRACT_TYPE=contract_type,
        FLAG_OWN_CAR=owns_car,
        FLAG_OWN_REALTY=owns_realty,
        CNT_CHILDREN=children,
    )

    st.session_state["form_submitted"] = submitted
    return applicant


def render_prediction_result(applicant: CreditApplicant, artifact: dict[str, Any]) -> None:
    input_df = build_model_input(applicant, artifact)
    probability = float(predict_positive_probability(artifact["pipeline"], input_df)[0])
    risk_level = classify_risk(probability)

    st.subheader("Resultado da simulação")

    col_prob, col_risk = st.columns(2)
    col_prob.metric("Probabilidade estimada", format_probability(probability))
    col_risk.metric("Classificação de risco", risk_level)

    if risk_level == "low":
        st.success("Risco baixo nesta simulação.")
    elif risk_level == "medium":
        st.warning("Risco médio nesta simulação.")
    else:
        st.error("Risco alto nesta simulação.")

    st.caption(OBSERVATION)

    with st.expander("Dados enviados ao modelo"):
        st.json(applicant.model_dump())


def render_metrics(metrics: pd.DataFrame | None) -> None:
    st.subheader("Métricas do treino")

    if metrics is None:
        st.info(
            "Ainda não encontrei `reports/model_metrics.csv`. "
            "Rode `python src/models/train_model.py` para gerar as métricas."
        )
        return

    sort_columns = [column for column in ["pr_auc", "roc_auc"] if column in metrics.columns]
    if sort_columns:
        metrics = metrics.sort_values(sort_columns, ascending=False)

    display_columns = [
        column
        for column in [
            "model",
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "f1_score",
            "sample_size",
            "test_rows",
        ]
        if column in metrics.columns
    ]

    best_model = metrics.iloc[0]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Melhor modelo", str(best_model.get("model", "N/A")))
    if "pr_auc" in best_model:
        metric_cols[1].metric("PR-AUC", f"{best_model['pr_auc']:.4f}")
    if "roc_auc" in best_model:
        metric_cols[2].metric("ROC-AUC", f"{best_model['roc_auc']:.4f}")
    if "recall" in best_model:
        metric_cols[3].metric("Recall", f"{best_model['recall']:.4f}")

    st.dataframe(metrics[display_columns], use_container_width=True, hide_index=True)

    if "sample_size" in metrics.columns and metrics["sample_size"].notna().any():
        st.caption(
            "As métricas exibidas foram carregadas do CSV local. "
            "Se o treino usou `--sample-size`, os valores refletem essa amostra."
        )


def render_project_summary() -> None:
    st.subheader("Resumo do projeto")
    st.write(
        "Este dashboard faz parte de um projeto de portfólio para prever risco de "
        "inadimplência usando dados tabulares da competição Home Credit Default Risk. "
        "O pipeline cobre preparação de dados, criação de features financeiras, "
        "comparação de modelos, explicabilidade e uma camada simples de deploy."
    )
    st.write(
        "A predição abaixo é uma simulação local. Ela não deve ser usada para aprovar, "
        "negar ou precificar crédito real."
    )


def main() -> None:
    st.title("Credit Risk Prediction")
    st.caption("Dashboard local para simular risco de inadimplência em crédito.")

    artifact = get_model_artifact()
    metrics = load_metrics()

    render_project_summary()

    if artifact is None:
        st.error(
            f"Modelo não encontrado em `{DEFAULT_MODEL_PATH}`. "
            "Treine o modelo antes de usar o dashboard."
        )
        st.code("python src/models/train_model.py --sample-size 5000", language="powershell")
        return

    model_name = artifact.get("metadata", {}).get("best_model", "modelo salvo")
    st.info(f"Modelo carregado: {model_name}")

    prediction_tab, metrics_tab = st.tabs(["Simulação", "Métricas"])

    with prediction_tab:
        left, right = st.columns([1.15, 1])
        with left:
            applicant = build_applicant_from_form()
        with right:
            if st.session_state.get("form_submitted", False):
                render_prediction_result(applicant, artifact)
            else:
                st.subheader("Resultado da simulação")
                st.write("Preencha os campos e clique em **Calcular risco**.")
                st.caption(OBSERVATION)

    with metrics_tab:
        render_metrics(metrics)


if __name__ == "__main__":
    main()
