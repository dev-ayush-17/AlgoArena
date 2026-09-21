# Backend Implementation Guide — 02: Loading Artifacts & Schemas

This document covers loading pickled ML model artifacts into application memory once at server startup, maintaining them in global state across requests, and defining strict Pydantic schemas that mirror the **Online Shoppers Purchasing Intention** dataset.

---

## Step 2.1: In-Memory Model & Transformer Singleton Service

### What & Why
Deserializing ML models (`joblib.load`) involves disk I/O and unpickling CPU overhead. Loading model artifacts per-request increases latency from milliseconds to hundreds of milliseconds or seconds, destroying serving performance. 

Loading all 5 model `.pkl` files plus `scaler.pkl` and `encoder.pkl` **once at startup** into an in-memory dictionary/singleton makes inference instantaneous across all HTTP requests.

### Code
Write `backend/services/model_service.py`:

```python
import os
import joblib
import logging
from typing import Dict, Any, Optional
from backend.config import settings

logger = logging.getLogger("backend.model_service")

class ModelContainer:
    """
    Singleton container holding all serialized models and preprocessors in memory.
    """
    def __init__(self):
        self.scaler: Optional[Any] = None
        self.encoder: Optional[Any] = None
        self.models: Dict[str, Any] = {}
        self.is_loaded: bool = False

    def load_artifacts(self) -> None:
        """Loads fitted transformers and 5 ML models into memory."""
        artifacts_dir = settings.ARTIFACTS_DIR
        logger.info(f"Loading ML artifacts from: {artifacts_dir}")

        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Artifacts directory missing: {artifacts_dir}")

        # 1. Load Preprocessors
        scaler_path = artifacts_dir / settings.SCALER_FILENAME
        encoder_path = artifacts_dir / settings.ENCODER_FILENAME

        if not scaler_path.exists() or not encoder_path.exists():
            raise FileNotFoundError("scaler.pkl or encoder.pkl missing from artifacts directory!")

        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)
        logger.info("Successfully loaded scaler.pkl and encoder.pkl")

        # 2. Load Models
        loaded_count = 0
        for model_key, filename in settings.MODEL_FILES.items():
            model_path = artifacts_dir / filename
            if not model_path.exists():
                logger.warning(f"Model artifact missing for '{model_key}': {model_path}. Skipping.")
                continue

            try:
                model = joblib.load(model_path)
                self.models[model_key] = model
                loaded_count += 1
                logger.info(f"Loaded model artifact: '{model_key}' from {filename}")
            except Exception as e:
                logger.error(f"Failed to unpickle model '{model_key}': {str(e)}")

        if loaded_count == 0:
            raise RuntimeError("No model artifacts were successfully loaded into memory!")

        self.is_loaded = True
        logger.info(f"Artifact initialization complete. Total models ready: {loaded_count}")

# Singleton Instance
model_container = ModelContainer()
```

### Expected Output & How to Verify It
Run a quick python check to test artifact loading:

```bash
python -c "from backend.services.model_service import model_container; model_container.load_artifacts(); print('Loaded models:', list(model_container.models.keys()))"
```

**Expected terminal output:**
```text
[INFO] Successfully loaded scaler.pkl and encoder.pkl
[INFO] Loaded model artifact: 'logistic_regression' from logistic_regression.pkl
[INFO] Loaded model artifact: 'knn' from knn.pkl
[INFO] Loaded model artifact: 'svm' from svm.pkl
[INFO] Loaded model artifact: 'naive_bayes' from naive_bayes.pkl
Loaded models: ['logistic_regression', 'knn', 'svm', 'naive_bayes']
```

*(Note: `decision_tree.pkl` will log a warning if missing until re-serialized in training).*

### Common Pitfalls
1. **Loading per Request:** Calling `joblib.load("model.pkl")` inside the endpoint function `def predict(...)`. This reads from disk on every single HTTP POST, causing severe latency spikes and potential file lock issues under concurrent load.
2. **Silent Unpickling Failures:** Failing to log missing or corrupted `.pkl` files, leaving the container in an inconsistent state where predictions break at runtime.

### Git Checkpoint
- **Branch:** `backend/feature-model-loading`
- **Commit:** `feat(model-loading): implement in-memory artifact singleton loader`

---

## Step 2.2: Lifespan Manager in Main Application

### What & Why
FastAPI provides the `lifespan` context manager mechanism (replacing legacy `on_event("startup")`) to run initialization code before accepting HTTP traffic. If artifact loading fails, the server fails fast at boot time rather than returning runtime 500 errors to users.

### Code
Update `backend/main.py`:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.services.model_service import model_container
from backend.routers import health

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML artifacts into memory
    print("[STARTUP] Initializing ML Artifact Container...")
    try:
        model_container.load_artifacts()
    except Exception as e:
        print(f"[FATAL] Failed to load ML artifacts: {e}")
        raise e
    yield
    # Shutdown: Clean up resources if needed
    print("[SHUTDOWN] Cleaning up resources...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=settings.API_V1_STR)
```

Update `backend/routers/health.py` to expose model status:

```python
from fastapi import APIRouter, Response, status
from backend.config import settings
from backend.services.model_service import model_container

router = APIRouter(tags=["Health"])

@router.get("/health")
def health_check(response: Response):
    loaded_models = list(model_container.models.keys())
    is_healthy = model_container.is_loaded and len(loaded_models) > 0
    
    if not is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "unhealthy",
            "models_loaded": len(loaded_models),
            "error": "Model container not fully initialized."
        }
        
    return {
        "status": "healthy",
        "models_loaded": len(loaded_models),
        "models": loaded_models,
        "transformers_loaded": model_container.scaler is not None and model_container.encoder is not None
    }
```

### Expected Output & How to Verify It
Start server and curl health endpoint:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

**Expected JSON response:**
```json
{
  "status": "healthy",
  "models_loaded": 4,
  "models": [
    "logistic_regression",
    "knn",
    "svm",
    "naive_bayes"
  ],
  "transformers_loaded": true
}
```

### Common Pitfalls
1. **Catching Lifespan Exceptions Silently:** Suppressing errors during startup allows Uvicorn to boot with `model_container.models = {}`. Every subsequent call to `/predict` will crash with unhandled exceptions.

### Git Checkpoint
- **Branch:** `backend/feature-model-loading`
- **Commit:** `feat(model-loading): wire lifespan startup artifact loading and health status`

---

## Step 2.3: Pydantic Input/Output Feature Schemas

### What & Why
The API input schema must validate raw feature inputs before passing them to the preprocessing pipeline. The **Online Shoppers Purchasing Intention** dataset has 10 numerical features, 5 integer categoricals, 2 string categoricals (`Month`, `VisitorType`), and 1 boolean flag (`Weekend`). Pydantic v2 schemas enforce field constraints, types, and defaults.

### Code
Write `backend/schemas/predict.py`:

```python
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional

class FeatureInput(BaseModel):
    # Numerical Features (10)
    Administrative: int = Field(default=0, ge=0, le=30, description="Number of administrative pages visited")
    Administrative_Duration: float = Field(default=0.0, ge=0.0, le=5000.0, description="Time spent on administrative pages (sec)")
    Informational: int = Field(default=0, ge=0, le=30, description="Number of informational pages visited")
    Informational_Duration: float = Field(default=0.0, ge=0.0, le=5000.0, description="Time spent on informational pages (sec)")
    ProductRelated: int = Field(default=1, ge=0, le=1000, description="Number of product-related pages visited")
    ProductRelated_Duration: float = Field(default=10.0, ge=0.0, le=70000.0, description="Time spent on product pages (sec)")
    BounceRates: float = Field(default=0.0, ge=0.0, le=1.0, description="Percentage of visitors leaving immediately")
    ExitRates: float = Field(default=0.02, ge=0.0, le=1.0, description="Percentage of pageviews that were last in session")
    PageValues: float = Field(default=0.0, ge=0.0, le=400.0, description="Google Analytics page value index")
    SpecialDay: float = Field(default=0.0, ge=0.0, le=1.0, description="Closeness of site visit to a special day (0.0 to 1.0)")
    
    # Categorical Features (7)
    Month: str = Field(default="May", description="Month of session (Feb, Mar, May, June, Jul, Aug, Sep, Oct, Nov, Dec)")
    OperatingSystems: int = Field(default=1, ge=1, le=8, description="Operating System ID")
    Browser: int = Field(default=2, ge=1, le=13, description="Browser ID")
    Region: int = Field(default=1, ge=1, le=9, description="Geographic Region ID")
    TrafficType: int = Field(default=2, ge=1, le=20, description="Traffic Source Type ID")
    VisitorType: str = Field(default="Returning_Visitor", description="Visitor type (Returning_Visitor, New_Visitor, Other)")
    Weekend: int = Field(default=0, ge=0, le=1, description="Session on weekend (0=No, 1=Yes)")

    @field_validator('Month')
    @classmethod
    def validate_month(cls, v: str) -> str:
        valid_months = {"Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}
        if v not in valid_months:
            raise ValueError(f"Invalid Month '{v}'. Must be one of: {sorted(list(valid_months))}")
        return v

    @field_validator('VisitorType')
    @classmethod
    def validate_visitor_type(cls, v: str) -> str:
        valid_types = {"Returning_Visitor", "New_Visitor", "Other"}
        if v not in valid_types:
            raise ValueError(f"Invalid VisitorType '{v}'. Must be one of: {sorted(list(valid_types))}")
        return v


class PredictionRequest(BaseModel):
    features: FeatureInput


class SingleModelPrediction(BaseModel):
    model_name: str
    model_key: str
    prediction: int
    prediction_label: str
    confidence: float
    inference_time_ms: float


class ConsensusSummary(BaseModel):
    majority_prediction: int
    majority_label: str
    agreement_ratio: float
    models_agreeing: int
    models_total: int


class PredictionResponse(BaseModel):
    predictions: List[SingleModelPrediction]
    consensus: ConsensusSummary
    input_echo: Dict[str, Any]
```

### Expected Output & How to Verify It
Test schema validation using Python CLI:

```bash
python -c "from backend.schemas.predict import FeatureInput; f = FeatureInput(Month='May', VisitorType='New_Visitor'); print('Valid Schema:', f.model_dump())"
```

**Expected terminal output:**
```text
Valid Schema: {'Administrative': 0, 'Administrative_Duration': 0.0, ..., 'Month': 'May', 'VisitorType': 'New_Visitor', 'Weekend': 0}
```

### Common Pitfalls
1. **Discrepancy Between Training & Schema Feature Names:** Using lower-case feature names (e.g. `administrative` or `bounce_rates`) when `preprocessing.py` and `StandardScaler` were fitted on exact capitalized names (`Administrative`, `BounceRates`).
2. **Missing Input Bounds:** Failing to place `ge` (greater than or equal) and `le` (less than or equal) constraints on numerical values like `BounceRates` allows invalid values (e.g., `BounceRates = 5.0`) to slip into ML inference.

### Git Checkpoint
- **Branch:** `backend/feature-model-loading`
- **Commit:** `feat(model-loading): define pydantic schemas for feature inputs and prediction outputs`

---

## Step 2.4: Safe Feature Preprocessing Protocol

### What & Why
> **CRITICAL ML SERVING RULE:** Live request data MUST be transformed using the exact saved `scaler.pkl` and `encoder.pkl` fitted during training. NEVER fit a scaler/encoder on incoming HTTP request data (`fit_transform`).

Inference preprocessing transforms raw input features into the exact matrix format expected by `model.predict()`.

### Code
Write `backend/services/preprocessing_service.py`:

```python
import pandas as pd
import numpy as np
from typing import Tuple
from backend.services.model_service import model_container
from backend.schemas.predict import FeatureInput

# Column Definitions matching model_training/preprocessing.py
NUM_COLS = [
    'Administrative', 'Administrative_Duration', 'Informational', 
    'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
]
INT_CAT_COLS = ['OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Weekend']
OHE_CAT_COLS = ['Month', 'VisitorType']


def transform_features_for_inference(feature_input: FeatureInput) -> pd.DataFrame:
    """
    Transforms raw Pydantic FeatureInput into the processed DataFrame feature matrix
    expected by serialized scikit-learn models. Reuses saved scaler and encoder.
    """
    if not model_container.is_loaded:
        raise RuntimeError("Model container is not initialized.")

    scaler = model_container.scaler
    encoder = model_container.encoder

    # 1. Convert input Pydantic model to dict -> DataFrame (1 row)
    raw_dict = feature_input.model_dump()
    raw_df = pd.DataFrame([raw_dict])

    # 2. Transform Categorical Features using saved OneHotEncoder (transform ONLY)
    encoded_cats = encoder.transform(raw_df[OHE_CAT_COLS])
    encoded_feature_names = encoder.get_feature_names_out(OHE_CAT_COLS)
    df_encoded = pd.DataFrame(encoded_cats, columns=encoded_feature_names, index=raw_df.index)

    # 3. Construct unified feature DataFrame
    feature_cols = NUM_COLS + INT_CAT_COLS
    X_raw = pd.concat([raw_df[feature_cols], df_encoded], axis=1)

    # 4. Scale Numerical Features using saved StandardScaler (transform ONLY)
    X_processed = X_raw.copy()
    X_processed[NUM_COLS] = scaler.transform(X_raw[NUM_COLS])

    return X_processed
```

### Expected Output & How to Verify It
Run a test script unpickling models and calling preprocessing:

```bash
python -c "from backend.services.model_service import model_container; model_container.load_artifacts(); from backend.schemas.predict import FeatureInput; from backend.services.preprocessing_service import transform_features_for_inference; X = transform_features_for_inference(FeatureInput()); print('Transformed Matrix Shape:', X.shape)"
```

**Expected terminal output:**
```text
Transformed Matrix Shape: (1, 28)
```

### Common Pitfalls
1. **Calling `.fit_transform()` on Live Data:** Re-fitting `StandardScaler` on a single row sets the mean to that row's value and standard deviation to 0, producing NaN/zero values for every numerical feature and corrupting model outputs.

### Git Checkpoint
- **Branch:** `backend/feature-model-loading`
- **Commit:** `feat(model-loading): implement inference preprocessing service using saved transformers`
