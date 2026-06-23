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

O notebook `04_explainability.ipynb` carrega o melhor modelo salvo e gera uma análise global de importância das variáveis. A abordagem principal é SHAP. Quando o SHAP não for compatível com o modelo ou ficar pesado para executar, o notebook usa uma alternativa segura com importância nativa de modelos de árvore ou permutation importance.

Artefatos esperados:

- `reports/figures/shap_summary.png`;
- `reports/figures/shap_bar.png`.

Pontos a documentar depois da execução final:

- principais variáveis globais identificadas;
- método usado na explicação (`shap`, importância de árvore ou permutation importance);
- sinais que parecem plausíveis do ponto de vista de crédito;
- variáveis que podem exigir revisão por risco de proxy ou viés;
- limitações da análise.

Cuidados de interpretação:

- explicabilidade mostra como o modelo usa os dados, não causalidade;
- variáveis correlacionadas podem dividir ou distorcer importância;
- categorias criadas por one-hot encoding devem ser interpretadas no contexto da variável original;
- decisões de crédito exigem critérios auditáveis, possibilidade de contestação e avaliação de fairness;
- a base do Kaggle é adequada para estudo, mas não substitui validação com dados atuais de uma operação real.

## 8. Deploy local

A preencher com observações sobre API FastAPI e app Streamlit.

## 9. Próximos passos

- Validar com dados mais recentes.
- Ajustar threshold com apoio de uma regra de negócio.
- Monitorar drift de dados e performance.
- Documentar riscos éticos e limitações do uso em decisão de crédito.
