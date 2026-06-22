# Relatório Final - Credit Risk Prediction

## 1. Contexto

Este projeto tem como objetivo construir um pipeline de Machine Learning para estimar risco de inadimplência em crédito a partir de dados tabulares.

## 2. Dataset

A preencher com:

- origem dos dados;
- número de linhas e colunas;
- definição da variável alvo;
- principais cuidados de qualidade;
- possíveis limitações ou vieses.

## 3. Análise exploratória

A preencher após o notebook `01_eda.ipynb`.

Pontos mínimos:

- proporção de inadimplentes;
- dados ausentes;
- distribuição das principais variáveis;
- variáveis com possível vazamento de informação;
- relações relevantes com a variável alvo.

## 4. Feature engineering

A preencher após o notebook `02_feature_engineering.ipynb`.

Descrever:

- tratamento de nulos;
- encoding de variáveis categóricas;
- escala de variáveis numéricas;
- novas variáveis criadas, se houver.

## 5. Modelagem

A preencher após o notebook `03_modeling.ipynb`.

| Modelo | ROC-AUC | PR-AUC | Recall classe 1 | Precision classe 1 | F1 classe 1 |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | A preencher | A preencher | A preencher | A preencher | A preencher |
| Random Forest | A preencher | A preencher | A preencher | A preencher | A preencher |
| XGBoost | A preencher | A preencher | A preencher | A preencher | A preencher |
| LightGBM | A preencher | A preencher | A preencher | A preencher | A preencher |

## 6. Modelo escolhido

A preencher.

Justificar a escolha considerando métrica, estabilidade, interpretabilidade e custo dos erros.

## 7. Explicabilidade

A preencher após o notebook `04_explainability.ipynb`.

Incluir:

- principais variáveis globais;
- exemplos de explicação local;
- cuidados ao interpretar o modelo.

## 8. Deploy local

A preencher com observações sobre API FastAPI e app Streamlit.

## 9. Próximos passos

- Validar com dados mais recentes.
- Ajustar threshold com apoio de uma regra de negócio.
- Monitorar drift de dados e performance.
- Documentar riscos éticos e limitações do uso em decisão de crédito.
