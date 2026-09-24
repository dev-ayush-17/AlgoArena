"""`config.py` centralizes paths to serialized model artifacts (`/model_training/artifacts/`) and application settings."""

import os
import json
from pathlib import Path
from typing import Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator

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
    ENVIRONMENT: str = "development"
    PORT: int = 8000

    # CORS Configuration
    ALLOWED_ORIGINS: Union[str, list[str]] = ["*"]

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        if isinstance(v, list):
            return v
        return ["*"]

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

    model_config = SettingsConfigDict(case_sensitive=True, extra="ignore")

settings = Settings()
