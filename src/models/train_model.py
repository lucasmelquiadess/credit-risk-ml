from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.features.build_features import (  # noqa: E402
    build_features,
    build_preprocessor,
    split_features_target,
)
from src.models.evaluate_model import (  # noqa: E402
    evaluate_classifier,
    metrics_to_frame,
    predict_positive_probability,
    save_confusion_matrix_plot,
    save_precision_recall_curve_plot,
    save_roc_curve_plot,
)
from src.utils.config import (  # noqa: E402
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_DATA_PATH,
    FIGURES_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    ensure_directories,
)


def load_training_data(
    data_path: Path = DEFAULT_PROCESSED_DATA_PATH,
    sample_size: int | None = None,
    target_column: str = TARGET_COLUMN,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {data_path}. "
            "Run python src/data/make_dataset.py first."
        )

    df = pd.read_csv(data_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    if sample_size is not None and sample_size < len(df):
        if sample_size < df[target_column].nunique() * 2:
            raise ValueError("sample_size is too small for stratified sampling.")

        sampled_df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df[target_column],
            random_state=random_state,
        )
        df = sampled_df.reset_index(drop=True)

    return df


def get_candidate_models(
    random_state: int = RANDOM_STATE,
    scale_pos_weight: float | None = None,
) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        print("lightgbm is not installed. Skipping LightGBM.")

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
        )
    except ImportError:
        print("xgboost is not installed. Skipping XGBoost.")

    return models


def build_model_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def log_run_to_mlflow(
    model_name: str,
    metrics: dict[str, Any],
    sample_size: int | None,
) -> None:
    try:
        import mlflow
    except ImportError:
        print("mlflow is not installed. Skipping MLflow logging.")
        return

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("sample_size", sample_size)
        mlflow.log_metrics(
            {key: value for key, value in metrics.items() if isinstance(value, float)}
        )


def train_and_compare(
    data_path: Path = DEFAULT_PROCESSED_DATA_PATH,
    target_column: str = TARGET_COLUMN,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    test_size: float = TEST_SIZE,
    sample_size: int | None = None,
    threshold: float = 0.5,
    random_state: int = RANDOM_STATE,
    log_mlflow: bool = False,
) -> dict[str, Any]:
    ensure_directories()

    df = load_training_data(
        data_path=data_path,
        sample_size=sample_size,
        target_column=target_column,
        random_state=random_state,
    )
    df = build_features(df)
    X, y = split_features_target(df, target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    scale_pos_weight = negative_count / positive_count if positive_count else None

    metrics_by_model: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Pipeline] = {}
    probabilities_by_model: dict[str, Any] = {}

    for model_name, estimator in get_candidate_models(
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
    ).items():
        print(f"Training {model_name}...")
        pipeline = build_model_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        y_proba = predict_positive_probability(pipeline, X_test)
        metrics = evaluate_classifier(
            pipeline,
            X_test,
            y_test,
            threshold=threshold,
        )
        metrics["sample_size"] = sample_size
        metrics["dataset_rows"] = len(df)
        metrics["train_rows"] = len(X_train)
        metrics["test_rows"] = len(X_test)

        metrics_by_model[model_name] = metrics
        fitted_models[model_name] = pipeline
        probabilities_by_model[model_name] = y_proba

        if log_mlflow:
            log_run_to_mlflow(model_name, metrics, sample_size)

    ranking = metrics_to_frame(metrics_by_model)
    best_model_name = str(ranking.iloc[0]["model"])
    best_pipeline = fitted_models[best_model_name]
    best_y_proba = probabilities_by_model[best_model_name]

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(metrics_path, index=False)

    save_confusion_matrix_plot(
        y_test,
        best_y_proba,
        output_path=FIGURES_DIR / "confusion_matrix.png",
        threshold=threshold,
        title=f"Confusion Matrix - {best_model_name}",
    )
    save_roc_curve_plot(
        y_test,
        probabilities_by_model,
        output_path=FIGURES_DIR / "roc_curve.png",
    )
    save_precision_recall_curve_plot(
        y_test,
        probabilities_by_model,
        output_path=FIGURES_DIR / "precision_recall_curve.png",
    )

    artifact = {
        "pipeline": best_pipeline,
        "metadata": {
            "target_column": target_column,
            "feature_columns": list(X.columns),
            "best_model": best_model_name,
            "selection_metric": "pr_auc",
            "threshold": threshold,
            "test_size": test_size,
            "sample_size": sample_size,
            "random_state": random_state,
            "trained_at": datetime.now(UTC).isoformat(),
            "metrics": metrics_by_model[best_model_name],
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    print("\nModel comparison:")
    print(ranking.to_string(index=False))
    print(f"\nBest model by PR-AUC: {best_model_name}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved best model to {model_path}")
    print(f"Saved figures to {FIGURES_DIR}")

    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train credit risk models.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_PROCESSED_DATA_PATH)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
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
        sample_size=args.sample_size,
        threshold=args.threshold,
        log_mlflow=args.log_mlflow,
    )


if __name__ == "__main__":
    main()
