from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.config import (
    DEFAULT_PROCESSED_DATA_PATH,
    DEFAULT_RAW_DATA_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    ensure_directories,
)


def generate_sample_credit_data(
    n_samples: int = 500,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Create a deterministic synthetic dataset for smoke tests."""
    rng = np.random.default_rng(random_state)

    age = rng.integers(21, 70, size=n_samples)
    income = rng.normal(65000, 22000, size=n_samples).clip(18000, 180000)
    loan_amount = rng.normal(22000, 12000, size=n_samples).clip(1000, 90000)
    loan_term_months = rng.choice([12, 24, 36, 48, 60, 72], size=n_samples)
    interest_rate = rng.normal(0.14, 0.05, size=n_samples).clip(0.03, 0.35)
    employment_years = rng.normal(6, 4, size=n_samples).clip(0, 35)
    credit_history_years = rng.normal(8, 5, size=n_samples).clip(0, 40)
    existing_debt = rng.normal(12000, 9000, size=n_samples).clip(0, 90000)
    missed_payments_2y = rng.poisson(0.7, size=n_samples).clip(0, 8)
    has_mortgage = rng.binomial(1, 0.35, size=n_samples)
    loan_purpose = rng.choice(
        ["car", "home_improvement", "medical", "education", "debt_consolidation"],
        size=n_samples,
        p=[0.25, 0.2, 0.12, 0.18, 0.25],
    )

    debt_to_income = (loan_amount + existing_debt) / income
    score = (
        -2.2
        + 2.5 * debt_to_income
        + 4.0 * interest_rate
        + 0.35 * missed_payments_2y
        - 0.04 * employment_years
        - 0.03 * credit_history_years
        + rng.normal(0, 0.55, size=n_samples)
    )
    probability_default = 1 / (1 + np.exp(-score))
    default = rng.binomial(1, probability_default)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income.round(2),
            "loan_amount": loan_amount.round(2),
            "loan_term_months": loan_term_months,
            "interest_rate": interest_rate.round(4),
            "employment_years": employment_years.round(1),
            "credit_history_years": credit_history_years.round(1),
            "existing_debt": existing_debt.round(2),
            "missed_payments_2y": missed_payments_2y,
            "has_mortgage": has_mortgage,
            "loan_purpose": loan_purpose,
            "default": default,
        }
    )

    missing_income_mask = rng.random(n_samples) < 0.03
    missing_purpose_mask = rng.random(n_samples) < 0.02
    df.loc[missing_income_mask, "income"] = np.nan
    df.loc[missing_purpose_mask, "loan_purpose"] = np.nan

    return df


def load_raw_dataset(path: Path, target_column: str) -> tuple[pd.DataFrame, tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Add a CSV there or run with --use-sample."
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    initial_shape = df.shape

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

    return df, initial_shape


def clean_dataset(
    df: pd.DataFrame,
    target_column: str,
    missing_threshold: float = 0.60,
) -> tuple[pd.DataFrame, list[str]]:
    missing_share = df.isna().mean()
    columns_to_drop = [
        column
        for column, share in missing_share.items()
        if share > missing_threshold and column != target_column
    ]

    cleaned_df = (
        df.drop(columns=columns_to_drop)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return cleaned_df, columns_to_drop


def save_processed_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def print_dataset_summary(
    input_path: str,
    initial_shape: tuple[int, int],
    final_shape: tuple[int, int],
    removed_columns: list[str],
    target_distribution: pd.Series,
) -> None:
    target_percent = (target_distribution / target_distribution.sum() * 100).round(2)
    target_summary = pd.DataFrame(
        {
            "count": target_distribution,
            "percent": target_percent,
        }
    )

    print(f"Input file: {input_path}")
    print(f"Initial shape: {initial_shape}")
    print(f"Final shape: {final_shape}")
    print(f"Columns removed (>60% missing): {len(removed_columns)}")
    if removed_columns:
        print(f"Removed columns: {removed_columns}")
    print("\nTARGET distribution:")
    print(target_summary.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare credit risk dataset.")
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_DATA_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_PROCESSED_DATA_PATH)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--use-sample", action="store_true")
    parser.add_argument("--n-samples", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    if args.use_sample:
        df = generate_sample_credit_data(n_samples=args.n_samples)
        if args.target_column != "default":
            df = df.rename(columns={"default": args.target_column})
        initial_shape = df.shape
        source = f"synthetic sample ({args.n_samples} rows)"
    else:
        df, initial_shape = load_raw_dataset(args.raw_path, args.target_column)
        source = str(args.raw_path)

    df, removed_columns = clean_dataset(df, args.target_column)
    save_processed_dataset(df, args.output_path)
    print_dataset_summary(
        input_path=source,
        initial_shape=initial_shape,
        final_shape=df.shape,
        removed_columns=removed_columns,
        target_distribution=df[args.target_column].value_counts(dropna=False).sort_index(),
    )
    print(f"\nSaved processed dataset to {args.output_path}.")


if __name__ == "__main__":
    main()
