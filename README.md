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

O projeto usa o dataset da competição Home Credit Default Risk, disponível no Kaggle. O arquivo principal esperado é:

```text
data/raw/application_train.csv
```

A coluna alvo é `TARGET`, em que `1` representa clientes com dificuldade de pagamento e `0` representa clientes sem dificuldade registrada na base.

Por padrão, os dados brutos não são versionados no Git. Isso evita subir arquivos grandes, dados sensíveis ou credenciais como `kaggle.json` e `access_token`.

## Data preparation

A primeira etapa do projeto é transformar o arquivo bruto em uma versão processada, ainda sem modelagem. O script de preparação fica em `src/data/make_dataset.py` e faz um tratamento inicial simples:

- carrega `data/raw/application_train.csv`;
- valida se a coluna `TARGET` existe;
- remove colunas com mais de 60% de valores ausentes;
- remove linhas duplicadas;
- salva o resultado em `data/processed/credit_risk_processed.csv`;
- imprime um resumo com shape inicial, shape final, quantidade de colunas removidas e distribuição da variável alvo.

Os números desse resumo dependem do arquivo usado localmente, então eles não ficam fixados aqui no README. Para gerar o dataset processado, rode:

```powershell
python src/data/make_dataset.py
```

Se quiser usar outro arquivo CSV ou outro nome de alvo, use:

```powershell
python src/data/make_dataset.py --raw-path data/raw/application_train.csv --target-column TARGET
```

## Feature Engineering

A primeira camada de feature engineering cria razões financeiras simples a partir das colunas originais do Home Credit. Essas variáveis ajudam a comparar contratos em escala relativa, em vez de olhar apenas valores absolutos.

Features criadas em `src/features/build_features.py`:

- `CREDIT_INCOME_RATIO`: `AMT_CREDIT / AMT_INCOME_TOTAL`;
- `ANNUITY_INCOME_RATIO`: `AMT_ANNUITY / AMT_INCOME_TOTAL`;
- `EMPLOYED_AGE_RATIO`: `DAYS_EMPLOYED / DAYS_BIRTH`;
- `CREDIT_ANNUITY_RATIO`: `AMT_CREDIT / AMT_ANNUITY`.

Divisões por zero e valores infinitos são tratados como valores ausentes (`NaN`). A etapa de imputação do pipeline fica responsável por lidar com esses casos depois.

O notebook `notebooks/02_feature_engineering.ipynb` mostra as distribuições dessas variáveis e compara as medianas por `TARGET`. As conclusões devem ser lidas como hipóteses para modelagem, não como prova de causalidade ou de ganho de performance.

## Modeling

A etapa de modelagem treina e compara quatro modelos de classificação binária:

- Logistic Regression com `class_weight="balanced"`;
- Random Forest;
- LightGBM;
- XGBoost.

O script aplica as features financeiras, faz split treino/teste estratificado e usa um pipeline com imputação, escala para variáveis numéricas e one-hot encoding para variáveis categóricas.

Para rodar o treino completo:

```powershell
python src/models/train_model.py
```

Para validar o fluxo mais rapidamente em uma amostra:

```powershell
python src/models/train_model.py --sample-size 5000
```

Se quiser registrar as métricas no MLflow durante o treino:

```powershell
python src/models/train_model.py --log-mlflow
```

O treino salva:

- métricas em `reports/model_metrics.csv`;
- melhor modelo em `models/credit_risk_model.pkl`;
- matriz de confusão, curva ROC e curva precision-recall em `reports/figures`.

Como a base é desbalanceada, a comparação não depende só de acurácia. O projeto acompanha principalmente PR-AUC, ROC-AUC, precision, recall, F1 e matriz de confusão.

## Explainability

A etapa de explicabilidade carrega o melhor modelo salvo em `models/credit_risk_model.pkl` e analisa uma amostra do dataset processado. O notebook tenta usar SHAP para explicar a importância global das variáveis. Se o SHAP não for compatível com o modelo salvo ou ficar pesado para executar, o notebook usa uma alternativa segura com importância de árvore ou permutation importance.

Para rodar:

```powershell
jupyter notebook notebooks/04_explainability.ipynb
```

ou executar pela linha de comando:

```powershell
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/04_explainability.ipynb --ExecutePreprocessor.timeout=300
```

A etapa salva:

- `reports/figures/shap_summary.png`;
- `reports/figures/shap_bar.png`.

As interpretações dessa etapa devem ser lidas com cuidado. Importância de variável ajuda a entender o comportamento do modelo, mas não prova causalidade nem substitui uma análise de fairness, estabilidade temporal e impacto de negócio.

## API

A API em FastAPI carrega o modelo salvo em `models/credit_risk_model.pkl` e expõe uma rota simples para simular uma previsão individual. Antes de subir a API, treine um modelo pelo menos uma vez:

```powershell
python src/models/train_model.py --sample-size 5000
```

Para iniciar o servidor local:

```powershell
uvicorn app.api:app --reload
```

Depois acesse a documentação interativa:

```text
http://127.0.0.1:8000/docs
```

Exemplo de JSON para `POST /predict`:

```json
{
  "AMT_INCOME_TOTAL": 180000,
  "AMT_CREDIT": 600000,
  "AMT_ANNUITY": 28000,
  "DAYS_BIRTH": -14000,
  "DAYS_EMPLOYED": -2500,
  "NAME_CONTRACT_TYPE": "Cash loans",
  "CODE_GENDER": "F",
  "FLAG_OWN_CAR": "N",
  "FLAG_OWN_REALTY": "Y",
  "CNT_CHILDREN": 0
}
```

A resposta inclui `default_probability`, `risk_level` (`low`, `medium` ou `high`) e uma observação deixando claro que a previsão é experimental e faz parte de um projeto de portfólio.

## Streamlit Dashboard

O dashboard em Streamlit usa o mesmo modelo salvo em `models/credit_risk_model.pkl` e permite preencher uma solicitação simulada de crédito. Ele mostra a probabilidade estimada de inadimplência, a classificação de risco e as métricas disponíveis em `reports/model_metrics.csv`, quando esse arquivo existir.

Para rodar:

```powershell
streamlit run app/streamlit_app.py
```

Depois acesse a URL exibida no terminal, normalmente:

```text
http://localhost:8501
```

O app é apenas educacional e faz parte do projeto de portfólio. Ele não deve ser usado para decisão real de crédito.

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
python src/models/train_model.py
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
