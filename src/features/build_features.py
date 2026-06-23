from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_RATIOS = {
    "CREDIT_INCOME_RATIO": ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
    "ANNUITY_INCOME_RATIO": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
    "EMPLOYED_AGE_RATIO": ("DAYS_EMPLOYED", "DAYS_BIRTH"),
    "CREDIT_ANNUITY_RATIO": ("AMT_CREDIT", "AMT_ANNUITY"),
}


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator

    return result.replace([np.inf, -np.inf], np.nan)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        column
        for numerator, denominator in FEATURE_RATIOS.values()
        for column in (numerator, denominator)
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing columns for feature engineering: {missing_columns}")

    features_df = df.copy()

    for feature_name, (numerator, denominator) in FEATURE_RATIOS.items():
        features_df[feature_name] = _safe_divide(
            features_df[numerator],
            features_df[denominator],
        )

    return features_df


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    X = df.drop(columns=[target_column])
    y = df[target_column].astype(int)

    if y.nunique() != 2:
        raise ValueError("Target column must contain exactly two classes.")

    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, selector(dtype_include=np.number)),
            ("categorical", categorical_pipeline, selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
