import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    build_features,
    build_preprocessor,
    split_features_target,
)


def test_split_features_target_returns_X_and_y() -> None:
    df = pd.DataFrame(
        {
            "age": [30, 45, 52, 28],
            "loan_purpose": ["car", "medical", "car", "education"],
            "default": [0, 1, 0, 1],
        }
    )

    X, y = split_features_target(df, "default")

    assert "default" not in X.columns
    assert y.tolist() == [0, 1, 0, 1]


def test_split_features_target_raises_for_missing_target() -> None:
    df = pd.DataFrame({"age": [30, 45], "default_flag": [0, 1]})

    with pytest.raises(ValueError, match="Target column"):
        split_features_target(df, "default")


def test_preprocessor_handles_missing_numeric_and_categorical_values() -> None:
    df = pd.DataFrame(
        {
            "age": [30, np.nan, 52, 28],
            "income": [50000, 72000, np.nan, 41000],
            "loan_purpose": ["car", None, "car", "education"],
            "default": [0, 1, 0, 1],
        }
    )
    X, _ = split_features_target(df, "default")

    transformed = build_preprocessor().fit_transform(X)

    assert transformed.shape[0] == len(df)
    assert not np.isnan(np.asarray(transformed)).any()


def test_build_features_creates_expected_ratio_columns() -> None:
    df = pd.DataFrame(
        {
            "AMT_CREDIT": [200000.0],
            "AMT_INCOME_TOTAL": [100000.0],
            "AMT_ANNUITY": [10000.0],
            "DAYS_EMPLOYED": [-1000.0],
            "DAYS_BIRTH": [-20000.0],
        }
    )

    result = build_features(df)

    assert result.loc[0, "CREDIT_INCOME_RATIO"] == 2.0
    assert result.loc[0, "ANNUITY_INCOME_RATIO"] == 0.1
    assert result.loc[0, "EMPLOYED_AGE_RATIO"] == 0.05
    assert result.loc[0, "CREDIT_ANNUITY_RATIO"] == 20.0


def test_build_features_handles_division_by_zero() -> None:
    df = pd.DataFrame(
        {
            "AMT_CREDIT": [200000.0],
            "AMT_INCOME_TOTAL": [0.0],
            "AMT_ANNUITY": [0.0],
            "DAYS_EMPLOYED": [-1000.0],
            "DAYS_BIRTH": [0.0],
        }
    )

    result = build_features(df)

    ratio_columns = [
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "EMPLOYED_AGE_RATIO",
        "CREDIT_ANNUITY_RATIO",
    ]
    assert result[ratio_columns].isna().all().all()


def test_build_features_does_not_create_infinite_values() -> None:
    df = pd.DataFrame(
        {
            "AMT_CREDIT": [200000.0, np.inf],
            "AMT_INCOME_TOTAL": [100000.0, 0.0],
            "AMT_ANNUITY": [10000.0, 0.0],
            "DAYS_EMPLOYED": [-1000.0, np.inf],
            "DAYS_BIRTH": [-20000.0, 0.0],
        }
    )

    result = build_features(df)

    ratio_columns = [
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "EMPLOYED_AGE_RATIO",
        "CREDIT_ANNUITY_RATIO",
    ]
    assert not np.isinf(result[ratio_columns].to_numpy()).any()
