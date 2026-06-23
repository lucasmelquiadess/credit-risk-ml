import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = Path(os.getenv("MODEL_DIR", str(PROJECT_ROOT / "models")))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(PROJECT_ROOT / "reports")))
FIGURES_DIR = REPORTS_DIR / "figures"

DEFAULT_RAW_DATA_PATH = RAW_DATA_DIR / "application_train.csv"
DEFAULT_PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "credit_risk_processed.csv"
DEFAULT_MODEL_PATH = MODEL_DIR / "credit_risk_pipeline.joblib"
DEFAULT_METRICS_PATH = REPORTS_DIR / "model_metrics.json"

TARGET_COLUMN = os.getenv("TARGET_COLUMN", "TARGET")
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))


def ensure_directories() -> None:
    """Create local output folders used by the project."""
    for path in [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
