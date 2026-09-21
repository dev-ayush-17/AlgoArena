# Backend Implementation Guide — 05: Error Handling & Edge Cases

This document details error handling strategies and custom FastAPI exception handlers designed around machine-learning-specific failure modes, strictly cross-referenced against **PRD.md Section 12 (Edge Cases & Failure Modes)** and **Section 9 (API Contract)**.

---

## Step 5.1: Error Classification & Status Code Standards

### What & Why
ML APIs fail differently than traditional web applications. Aside from standard HTTP errors, ML services encounter domain-specific failure modes:
1. **Validation Failures (422 Unprocessable Entity):** Missing required feature fields or invalid data types (e.g. string passed for `BounceRates`).
2. **Out-of-Range Feature Inputs (400 Bad Request):** Numerical features outside physical limits (e.g. `BounceRates = 5.0` when max is 1.0) or unseen categorical options.
3. **Partial/Total Model Failure (500 Internal Server Error / 503 Service Unavailable):** Artifact loading crashes or runtime inference exceptions in a single model algorithm.

Defining explicit handlers ensures predictable JSON shapes for frontend error toasts and logging.

### Code
Create custom exception classes and response models in `backend/schemas/errors.py`:

```python
from pydantic import BaseModel
from typing import List, Optional, Any, Dict

class APIErrorDetail(BaseModel):
    loc: Optional[List[str]] = None
    msg: str
    type: str

class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[List[APIErrorDetail]] = None
    failed_models: Optional[List[str]] = None
```

Create exception handlers in `backend/routers/error_handlers.py`:

```python
import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

logger = logging.getLogger("backend.error_handlers")

class MLValidationException(Exception):
    """Raised when feature inputs break physical/domain boundaries."""
    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field = field

class ModelInferenceException(Exception):
    """Raised when a specific ML model fails during predict execution."""
    def __init__(self, message: str, model_key: str):
        self.message = message
        self.model_key = model_key


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Formats Pydantic 422 validation errors into standard project error shape."""
    details = []
    for error in exc.errors():
        loc = [str(x) for x in error.get("loc", [])]
        details.append({
            "loc": loc,
            "msg": error.get("msg", "Invalid value"),
            "type": error.get("type", "value_error")
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "message": "Input features failed payload validation.",
            "details": details
        }
    )

async def ml_validation_exception_handler(request: Request, exc: MLValidationException):
    """Formats 400 Bad Request domain boundary violations."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "invalid_feature_value",
            "message": exc.message,
            "field": exc.field
        }
    )

async def model_inference_exception_handler(request: Request, exc: ModelInferenceException):
    """Formats 500 Internal Server Error when ML inference execution fails."""
    logger.error(f"Inference error on model '{exc.model_key}': {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "prediction_error",
            "message": f"Model '{exc.model_key}' failed: {exc.message}",
            "failed_models": [exc.model_key]
        }
    )
```

Register handlers in `backend/main.py`:

```python
from fastapi.exceptions import RequestValidationError
from backend.routers.error_handlers import (
    MLValidationException, ModelInferenceException,
    validation_exception_handler, ml_validation_exception_handler,
    model_inference_exception_handler
)

# Add exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(MLValidationException, ml_validation_exception_handler)
app.add_exception_handler(ModelInferenceException, model_inference_exception_handler)
```

### Expected Output & How to Verify It
Send a malformed JSON request missing required fields:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"Administrative": "invalid_type"}}'
```

**Expected JSON Response (HTTP 422 Unprocessable Entity):**
```json
{
  "error": "validation_error",
  "message": "Input features failed payload validation.",
  "details": [
    {
      "loc": ["body", "features", "Administrative"],
      "msg": "Input should be a valid integer, unable to parse string as an integer",
      "type": "int_parsing"
    }
  ]
}
```

### Common Pitfalls
1. **Returning Default FastAPI 422 Format:** Default FastAPI 422 errors nest raw Python tuples in `loc` (`['body', 'features', 'Month']`), which breaks strict frontend error parsers expecting uniform JSON object lists.

### Git Checkpoint
- **Branch:** `backend/feature-error-handling`
- **Commit:** `feat(error-handling): implement standard exception handlers for 422, 400, and 500 error codes`

---

## Step 5.2: Handling Unseen Categoricals & Boundary Outliers

### What & Why
PRD.md §12 specifically mandates:
- **Out-of-range categorical values:** E.g., user submits `Month = "December"` (dataset uses `"Dec"`) or `VisitorType = "Robot"`.
- **Handling Strategy:** Saved `OneHotEncoder(handle_unknown='ignore')` safely outputs all zeros for unseen categories without throwing a Python runtime exception. However, validating against known categories before encoding gives the user immediate 400 validation feedback.

### Code
Update `backend/services/preprocessing_service.py` with boundary enforcement:

```python
from backend.schemas.predict import FeatureInput
from backend.routers.error_handlers import MLValidationException

def validate_domain_boundaries(features: FeatureInput) -> None:
    """Enforces PRD.md §12 domain limits on input feature values."""
    if features.BounceRates < 0.0 or features.BounceRates > 1.0:
        raise MLValidationException(
            message=f"BounceRates must be between 0.0 and 1.0, got {features.BounceRates}",
            field="BounceRates"
        )

    if features.ExitRates < 0.0 or features.ExitRates > 1.0:
        raise MLValidationException(
            message=f"ExitRates must be between 0.0 and 1.0, got {features.ExitRates}",
            field="ExitRates"
        )

    if features.SpecialDay < 0.0 or features.SpecialDay > 1.0:
        raise MLValidationException(
            message=f"SpecialDay must be between 0.0 and 1.0, got {features.SpecialDay}",
            field="SpecialDay"
        )
```

Wire check into `transform_features_for_inference()`:

```python
def transform_features_for_inference(feature_input: FeatureInput) -> pd.DataFrame:
    # Validate boundary bounds first
    validate_domain_boundaries(feature_input)
    # ... proceeding with scaling and encoding ...
```

### Expected Output & How to Verify It
Send an out-of-bounds request via `curl`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "BounceRates": 5.0
    }
  }'
```

**Expected JSON Response (HTTP 400 Bad Request):**
```json
{
  "error": "invalid_feature_value",
  "message": "BounceRates must be between 0.0 and 1.0, got 5.0",
  "field": "BounceRates"
}
```

### Common Pitfalls
1. **Crashing OneHotEncoder on Unseen Labels:** Using `OneHotEncoder(handle_unknown='error')` during training causes unhandled `ValueError` when an unseen string category is encountered at inference time. Using `handle_unknown='ignore'` combined with Pydantic validation prevents crashes.

### Git Checkpoint
- **Branch:** `backend/feature-error-handling`
- **Commit:** `feat(error-handling): add domain boundary checks for numerical and categorical features`

---

## Step 5.3: Partial Failure Resilience Architecture

### What & Why
If 1 out of 5 models fails during a live `POST /predict` call (for example, if a model file is missing or corrupted), the API should **not** return a complete 500 crash to the user. Instead, PRD.md §9 specifies partial failure recovery: return successful predictions alongside a `failed_models` array so the frontend UI can display available predictions gracefully.

### Code
Update `run_prediction_pipeline()` in `backend/services/model_service.py`:

```python
def run_prediction_pipeline_resilient(request: PredictionRequest) -> dict:
    """
    Executes prediction pipeline with partial failure support.
    Returns successfully calculated predictions and logs failed model keys.
    """
    X_processed = transform_features_for_inference(request.features)

    single_predictions: List[SingleModelPrediction] = []
    raw_preds: List[int] = []
    failed_models: List[str] = []

    for model_key, model in model_container.models.items():
        display_name = DISPLAY_NAMES.get(model_key, model_key.replace("_", " ").title())
        try:
            (prediction_arr), latency_ms = measure_inference_time_ms(model.predict, X_processed)
            pred_int = int(prediction_arr[0])
            raw_preds.append(pred_int)

            confidence = 0.5
            if hasattr(model, "predict_proba"):
                try:
                    proba_arr = model.predict_proba(X_processed)[0]
                    confidence = float(np.max(proba_arr))
                except Exception:
                    confidence = 0.5

            single_predictions.append(
                SingleModelPrediction(
                    model_name=display_name,
                    model_key=model_key,
                    prediction=pred_int,
                    prediction_label=CLASS_LABELS.get(pred_int, f"Class {pred_int}"),
                    confidence=round(confidence, 4),
                    inference_time_ms=latency_ms
                )
            )
        except Exception as e:
            logger.error(f"Model '{model_key}' prediction failed: {str(e)}")
            failed_models.append(model_key)

    if not single_predictions:
        raise RuntimeError("All models failed during inference execution!")

    majority_pred = int(max(set(raw_preds), key=raw_preds.count))
    models_agreeing = raw_preds.count(majority_pred)
    models_total = len(raw_preds)
    agreement_ratio = round(models_agreeing / models_total, 4)

    consensus = ConsensusSummary(
        majority_prediction=majority_pred,
        majority_label=CLASS_LABELS.get(majority_pred, f"Class {majority_pred}"),
        agreement_ratio=agreement_ratio,
        models_agreeing=models_agreeing,
        models_total=models_total
    )

    response_dict = {
        "predictions": single_predictions,
        "consensus": consensus,
        "input_echo": {
            "Administrative": request.features.Administrative,
            "ProductRelated": request.features.ProductRelated,
            "PageValues": request.features.PageValues,
            "Month": request.features.Month,
            "VisitorType": request.features.VisitorType,
            "Weekend": request.features.Weekend
        }
    }
    
    if failed_models:
        response_dict["failed_models"] = failed_models

    return response_dict
```

### Expected Output & How to Verify It
If one model `.pkl` file is deleted or disabled during testing, `POST /predict` returns HTTP 200 with partial predictions:

**Partial Failure Response:**
```json
{
  "predictions": [
    {
      "model_name": "Logistic Regression",
      "model_key": "logistic_regression",
      "prediction": 0,
      "prediction_label": "No Purchase",
      "confidence": 0.85,
      "inference_time_ms": 0.45
    }
  ],
  "consensus": {
    "majority_prediction": 0,
    "majority_label": "No Purchase",
    "agreement_ratio": 1.0,
    "models_agreeing": 1,
    "models_total": 1
  },
  "failed_models": ["knn", "svm"]
}
```

### Common Pitfalls
1. **Cascading Failures:** Letting an exception in `knn.predict()` crash the entire request handler prevents returning valid Logistic Regression or Naive Bayes predictions that computed successfully.

### Git Checkpoint
- **Branch:** `backend/feature-error-handling`
- **Commit:** `feat(error-handling): implement partial failure resilience for multi-model inference`
