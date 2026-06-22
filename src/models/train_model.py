from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.features.build_features import build_preprocessor, split_features_target
from src.models.evaluate_model import evaluate_classifier, metrics_to_frame
from src.utils.config import (
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    ensure_directories,
)


def get_candidate_models(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
    except ImportError:
        print("xgboost is not installed. Skipping XGBoost.")

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        print("lightgbm is not installed. Skipping LightGBM.")

    return models


def build_model_pipeline(
    estimator: Any,
    use_smote: bool = False,
    random_state: int = RANDOM_STATE,
) -> ImbPipeline:
    steps: list[tuple[str, Any]] = [("preprocess", build_preprocessor())]

    if use_smote:
        steps.append(("sampler", SMOTE(random_state=random_state)))

    steps.append(("model", estimator))
    return ImbPipeline(steps=steps)


def train_and_compare(
    data_path: Path,
    target_column: str = TARGET_COLUMN,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    use_smote: bool = False,
    log_mlflow: bool = False,
) -> dict[str, Any]:
    ensure_directories()

    df = pd.read_csv(data_path)
    X, y = split_features_target(df, target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    metrics_by_model: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, ImbPipeline] = {}

    for model_name, estimator in get_candidate_models(random_state).items():
        pipeline = build_model_pipeline(estimator, use_smote, random_state)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_classifier(pipeline, X_test, y_test)
        metrics_by_model[model_name] = metrics
        fitted_models[model_name] = pipeline

        if log_mlflow:
            log_run_to_mlflow(model_name, metrics, use_smote)

    ranking = metrics_to_frame(metrics_by_model)
    best_model_name = ranking.iloc[0]["model"]
    best_pipeline = fitted_models[best_model_name]

    artifact = {
        "pipeline": best_pipeline,
        "metadata": {
            "target_column": target_column,
            "feature_columns": list(X.columns),
            "best_model": best_model_name,
            "selection_metric": "pr_auc",
            "test_size": test_size,
            "random_state": random_state,
            "use_smote": use_smote,
            "trained_at": datetime.now(UTC).isoformat(),
            "metrics": metrics_by_model[best_model_name],
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "best_model": best_model_name,
        "selection_metric": "pr_auc",
        "models": metrics_by_model,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(ranking.to_string(index=False))
    print(f"\nSaved best model '{best_model_name}' to {model_path}.")
    print(f"Saved metrics to {metrics_path}.")

    return artifact


def log_run_to_mlflow(
    model_name: str,
    metrics: dict[str, Any],
    use_smote: bool,
) -> None:
    try:
        import mlflow
    except ImportError:
        print("mlflow is not installed. Skipping MLflow logging.")
        return

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("use_smote", use_smote)
        mlflow.log_metrics(
            {key: value for key, value in metrics.items() if isinstance(value, float)}
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train credit risk models.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_PROCESSED_DATA_PATH)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--use-smote", action="store_true")
    parser.add_argument("--log-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_compare(
        data_path=args.input_path,
        target_column=args.target_column,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        test_size=args.test_size,
        use_smote=args.use_smote,
        log_mlflow=args.log_mlflow,
    )


if __name__ == "__main__":
    main()
