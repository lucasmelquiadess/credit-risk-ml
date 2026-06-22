from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.models.evaluate_model import predict_positive_probability
from src.utils.config import DEFAULT_MODEL_PATH


def load_model_artifact(model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Train a model first."
        )
    return joblib.load(model_path)


def predict_records(
    records: list[dict[str, Any]],
    model_path: Path = DEFAULT_MODEL_PATH,
    threshold: float = 0.5,
) -> pd.DataFrame:
    artifact = load_model_artifact(model_path)
    pipeline = artifact["pipeline"]

    X = pd.DataFrame(records)
    probabilities = predict_positive_probability(pipeline, X)
    predictions = (probabilities >= threshold).astype(int)

    results = X.copy()
    results["predicted_default_probability"] = probabilities.round(6)
    results["predicted_default"] = predictions
    return results


def parse_input_json(input_json: str) -> list[dict[str, Any]]:
    payload = json.loads(input_json)
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    raise ValueError("--input-json must be a JSON object or a list of objects.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run credit risk predictions.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=0.5)

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-json")
    input_group.add_argument("--input-path", type=Path)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_json:
        records = parse_input_json(args.input_json)
    else:
        records = pd.read_csv(args.input_path).to_dict(orient="records")

    predictions = predict_records(records, args.model_path, args.threshold)
    print(predictions.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
