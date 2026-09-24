"""`config.py` centralizes paths to serialized model artifacts (`/model_training/artifacts/`) and application settings. Hardcoding absolute paths across router files creates fragile code that breaks on deployment or across different developer machines."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent
_ENV_ARTIFACTS = os.getenv("ARTIFACTS_DIR")
_DEFAULT_ARTIFACTS_DIR = Path(_ENV_ARTIFACTS).resolve() if _ENV_ARTIFACTS else (_PROJECT_ROOT / "model_training" / "artifacts")

class Settings(BaseSettings):
    # App information
    PROJECT_NAME: str = "Algorithm Arena API"
    VERSION: str = "1.0"
    API_V1_STR: str = "/api/v1"

    # Deployment Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    PORT: int = int(os.getenv("PORT", "8000"))

    # CORS Configuration
    ALLOWED_ORIGINS: list[str] = [
        origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()
    ]

    # Path Resolution
    BACKEND_DIR: Path = _BACKEND_DIR
    PROJECT_ROOT: Path = _PROJECT_ROOT
    ARTIFACTS_DIR: Path = _DEFAULT_ARTIFACTS_DIR

    # Artifacts Filenames
    SCALER_FILENAME: str = "scaler.pkl"
    ENCODER_FILENAME: str = "encoder.pkl"
    FEATURE_CONFIG_FILENAME: str = "feature_config.json"
    METRICS_FILENAME: str = "metrics.json"

    # Model keys and Filenames
    MODEL_FILES: dict[str, str] = {
        "logistic_regression": "logistic_regression.pkl",
        "knn": "knn.pkl",
        "svm": "svm.pkl",
        "decision_tree": "decision_tree.pkl",
        "naive_bayes": "naive_bayes.pkl"
    }

    model_config = SettingsConfigDict(case_sensitive=True)

settings = Settings()
