"""`config.py` centralizes paths to serialized model artifacts (`/model_training/artifacts/`) and application settings. Hardcoding absolute paths across router files creates fragile code that breaks on deployment or across different developer machines."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App information
    PROJECT_NAME: str = "Algorithm Arena API"
    VERSION: str = "1.0"
    API_V1_STR: str = "/api/v1"

    # Deployment Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    PORT: int = int(os.getenv("PORT", 8000))

     # CORS Configuration
    ALLOWED_ORIGINS: list = os.getenv(
        "ALLOWED_ORIGINS", 
        "*"
    ).split(",")

    # Path Resolution
    BACKEND_DIR: Path = Path(__file__).resolve().parent
    PROJECT_ROOT: Path = BACKEND_DIR.parent

    # env
    env_artifacts = os.getenv("ARTIFACTS_DIR")
    if env_artifacts:
        ARTIFACTS_DIR: Path = Path(env_artifacts).resolve()
    else:
        ARTIFACTS_DIR: Path = PROJECT_ROOT / "model_training" / "artifacts"

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
        "naive_bayes": "naive_bayes.pkl"
    }

    class Config:
        case_sensitive = True

settings = Settings()
