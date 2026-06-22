import numpy as np
import pandas as pd
import pytest

from src.features.build_features import build_preprocessor, split_features_target


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
