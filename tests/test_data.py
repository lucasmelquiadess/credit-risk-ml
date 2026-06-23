from pathlib import Path

import pandas as pd
import pytest

from src.data.make_dataset import (
    clean_dataset,
    load_raw_dataset,
    save_processed_dataset,
)


def test_clean_dataset_removes_sparse_columns_and_duplicates() -> None:
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2],
            "TARGET": [0, 0, 1],
            "mostly_missing": [None, None, "x"],
            "useful_feature": [10, 10, 20],
        }
    )

    cleaned_df, removed_columns = clean_dataset(
        df,
        target_column="TARGET",
        missing_threshold=0.60,
    )

    assert removed_columns == ["mostly_missing"]
    assert "mostly_missing" not in cleaned_df.columns
    assert cleaned_df.shape == (2, 3)


def test_clean_dataset_keeps_target_even_if_it_has_missing_values() -> None:
    df = pd.DataFrame(
        {
            "TARGET": [0, None, None],
            "feature": [1, 2, 3],
            "mostly_missing": [None, None, "x"],
        }
    )

    cleaned_df, removed_columns = clean_dataset(
        df,
        target_column="TARGET",
        missing_threshold=0.60,
    )

    assert "TARGET" in cleaned_df.columns
    assert "TARGET" not in removed_columns
    assert removed_columns == ["mostly_missing"]


def test_load_raw_dataset_validates_target_column(tmp_path: Path) -> None:
    raw_path = tmp_path / "application_train.csv"
    pd.DataFrame({"SK_ID_CURR": [1, 2], "OTHER_TARGET": [0, 1]}).to_csv(
        raw_path,
        index=False,
    )

    with pytest.raises(ValueError, match="Target column 'TARGET' was not found"):
        load_raw_dataset(raw_path, target_column="TARGET")


def test_load_raw_dataset_strips_column_names(tmp_path: Path) -> None:
    raw_path = tmp_path / "application_train.csv"
    pd.DataFrame({" TARGET ": [0, 1], "feature": [10, 20]}).to_csv(
        raw_path,
        index=False,
    )

    df, initial_shape = load_raw_dataset(raw_path, target_column="TARGET")

    assert initial_shape == (2, 2)
    assert "TARGET" in df.columns


def test_save_processed_dataset_creates_parent_directory(tmp_path: Path) -> None:
    output_path = tmp_path / "processed" / "credit_risk_processed.csv"
    df = pd.DataFrame({"TARGET": [0, 1], "feature": [10, 20]})

    save_processed_dataset(df, output_path)

    assert output_path.exists()
    saved_df = pd.read_csv(output_path)
    assert saved_df.equals(df)
