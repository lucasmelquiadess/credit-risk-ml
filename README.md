# Credit Risk Prediction with Machine Learning

Projeto de portfólio para construir um pipeline end-to-end de Machine Learning aplicado a risco de crédito. A ideia é prever a probabilidade de inadimplência de um cliente a partir de dados tabulares, com uma organização próxima do que eu usaria em um projeto profissional pequeno: ingestão de dados, tratamento de variáveis, treino, avaliação, explicabilidade e uma camada simples de deploy.

Este repositório ainda está na fase inicial. Os scripts já rodam localmente e existe um dataset sintético apenas para testar o fluxo, mas os resultados finais devem ser preenchidos somente depois do treino com um dataset real.

## Objetivo

Treinar e comparar modelos de classificação binária para estimar risco de inadimplência. O foco do projeto não é só obter uma métrica alta, mas mostrar um processo cuidadoso:

- análise exploratória antes da modelagem;
- tratamento de dados ausentes;
- separação entre treino e teste;
- feature engineering com pipeline reproduzível;
- comparação entre modelos lineares, ensemble e boosting;
- atenção a desbalanceamento de classes;
- avaliação com métricas adequadas para risco de crédito;
- explicabilidade com SHAP;
- API e app simples para consumo do modelo.

## Dataset esperado

O projeto espera um arquivo CSV em `data/raw/credit_data.csv`, com uma coluna alvo binária chamada `default`.

Exemplo de formato:

| age | income | loan_amount | loan_purpose | default |
| --- | --- | --- | --- | --- |
| 35 | 72000 | 18000 | car | 0 |
| 47 | 41000 | 25000 | debt_consolidation | 1 |

Se o dataset real usar outro nome para a coluna alvo, passe o nome com `--target-column`.

Por padrão, os dados brutos não são versionados no Git. Isso evita subir dados privados, arquivos grandes ou credenciais como `kaggle.json`.

## Como rodar localmente

Use Python 3.12.

```powershell
cd credit-risk-ml
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Para validar o pipeline sem dataset real, gere uma base sintética pequena:

```powershell
python -m src.data.make_dataset --use-sample
```

Treine os modelos:

```powershell
python -m src.models.train_model --input-path data/processed/credit_risk_dataset.csv --target-column default
```

Faça uma predição pela linha de comando:

```powershell
$record = '{\"age\":35,\"income\":72000,\"loan_amount\":18000,\"loan_term_months\":36,\"interest_rate\":0.12,\"employment_years\":6,\"credit_history_years\":8,\"existing_debt\":9000,\"missed_payments_2y\":0,\"has_mortgage\":1,\"loan_purpose\":\"car\"}'
python -m src.models.predict_model --input-json $record
```

Suba a API:

```powershell
uvicorn app.api:app --reload
```

Suba o app em Streamlit:

```powershell
streamlit run app/streamlit_app.py
```

Execute os testes:

```powershell
pytest
```

Cheque estilo do código:

```powershell
ruff check .
```

## Estrutura do projeto

```text
credit-risk-ml/
├── app/                  # API FastAPI e interface Streamlit
├── data/                 # Dados brutos, intermediários e processados
├── models/               # Artefatos treinados localmente
├── notebooks/            # Exploração, features, modelagem e SHAP
├── reports/              # Relatórios e figuras
├── src/                  # Código reutilizável do pipeline
└── tests/                # Testes automatizados
```

## Métricas que serão acompanhadas

Como o problema é de risco de crédito, acurácia sozinha não é suficiente. A avaliação deve olhar principalmente para a capacidade de encontrar inadimplentes sem gerar falsos positivos demais.

| Modelo | ROC-AUC | PR-AUC | Recall da classe 1 | Precision da classe 1 | F1 da classe 1 | Threshold |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | A preencher | A preencher | A preencher | A preencher | A preencher | A preencher |
| Random Forest | A preencher | A preencher | A preencher | A preencher | A preencher | A preencher |
| XGBoost | A preencher | A preencher | A preencher | A preencher | A preencher | A preencher |
| LightGBM | A preencher | A preencher | A preencher | A preencher | A preencher | A preencher |

## Plano de análise

1. Entender a distribuição da variável alvo.
2. Verificar dados ausentes, outliers e possíveis vazamentos de informação.
3. Criar um pipeline com imputação, escala para variáveis numéricas e one-hot encoding para variáveis categóricas.
4. Treinar uma baseline simples antes de modelos mais fortes.
5. Comparar modelos com foco em PR-AUC, recall e custo dos erros.
6. Ajustar hiperparâmetros com Optuna quando a baseline estiver estável.
7. Explicar o modelo escolhido com SHAP.
8. Publicar uma API simples e uma interface Streamlit para demonstrar o uso.

## Resultados

Esta seção será preenchida depois do treino com o dataset real.

### Melhor modelo

A preencher.

### Principais variáveis

A preencher após análise de importância e SHAP.

### Limitações

A preencher com base no dataset usado, nos vieses encontrados e no comportamento do modelo.

## Observações

O dataset sintético gerado por `src.data.make_dataset` serve apenas para testar se a estrutura funciona. Ele não deve ser usado para tirar conclusões sobre crédito, risco ou performance real de modelos.
