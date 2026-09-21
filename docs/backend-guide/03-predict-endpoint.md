# Backend Implementation Guide — 03: Predict Endpoint

This document guides you through building the core inference handler: `POST /api/v1/predict`. The endpoint accepts a single feature payload, transforms it using saved preprocessing artifacts, executes all loaded ML models side-by-side, measures high-precision inference latencies, and returns model outputs alongside consensus summary metrics.

---

## Step 3.1: High-Precision Latency Measurement Utility

### What & Why
To capture meaningful latency trade-offs between models (e.g. Logistic Regression @ 0.04ms vs KNN @ 14ms), inference time must be measured using monotonic microsecond timers (`time.perf_counter()`). `time.time()` lacks precision on Windows and can drift.

### Code
Write `backend/utils/timing.py`:

```python
import time
from typing import Tuple, Any, Callable

def measure_inference_time_ms(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    """
    Executes a callable function and measures wall-clock execution time in milliseconds.
    
    Returns:
        Tuple containing (function_result, elapsed_time_in_ms)
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    
    elapsed_ms = (end_time - start_time) * 1000.0
    return result, round(elapsed_ms, 4)
```

### Expected Output & How to Verify It
Verify utility accuracy using a sleep mock:

```bash
python -c "from backend.utils.timing import measure_inference_time_ms; import time; res, ms = measure_inference_time_ms(time.sleep, 0.01); print(f'Measured {ms} ms for 10ms sleep')"
```

**Expected output:**
```text
Measured 10.152 ms for 10ms sleep
```

### Common Pitfalls
1. **Using `time.time()`:** On Windows operating systems, `time.time()` resolution is ~15.6ms, rendering sub-millisecond model timings completely zero.

### Git Checkpoint
- **Branch:** `backend/feature-predict-endpoint`
- **Commit:** `feat(predict-endpoint): add microsecond precision timing utility`

---

## Step 3.2: Prediction Orchestration & Consensus Business Logic

### What & Why
The prediction service coordinates feature transformation, loops through all loaded models, handles probability confidence extraction, and calculates majority vote metrics across models.

### Code
Update `backend/services/model_service.py` to add `predict_all()` method:

```python
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from backend.services.model_service import model_container
from backend.services.preprocessing_service import transform_features_for_inference
from backend.schemas.predict import (
    PredictionRequest, PredictionResponse, SingleModelPrediction, ConsensusSummary
)
from backend.utils.timing import measure_inference_time_ms

# Display Names & Class Labels for Online Shoppers Dataset
DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "knn": "K-Nearest Neighbors",
    "svm": "Support Vector Machine",
    "decision_tree": "Decision Tree",
    "naive_bayes": "Naive Bayes"
}

CLASS_LABELS = {
    0: "No Purchase",
    1: "Purchase Completed"
}


def run_prediction_pipeline(request: PredictionRequest) -> PredictionResponse:
    """
    Orchestrates input feature transformation, multi-model execution,
    latency benchmarking, and consensus aggregation.
    """
    # 1. Transform raw features using saved scaler and encoder
    X_processed = transform_features_for_inference(request.features)

    single_predictions: List[SingleModelPrediction] = []
    raw_preds: List[int] = []

    # 2. Iterate through loaded models
    for model_key, model in model_container.models.items():
        display_name = DISPLAY_NAMES.get(model_key, model_key.replace("_", " ").title())

        # Measure predict latency
        (prediction_arr), latency_ms = measure_inference_time_ms(model.predict, X_processed)
        pred_int = int(prediction_arr[0])
        raw_preds.append(pred_int)

        # Extract confidence score (probability of predicted class)
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

    # 3. Consensus & Agreement Metrics
    if raw_preds:
        majority_pred = int(max(set(raw_preds), key=raw_preds.count))
        models_agreeing = raw_preds.count(majority_pred)
        models_total = len(raw_preds)
        agreement_ratio = round(models_agreeing / models_total, 4)
    else:
        majority_pred = 0
        models_agreeing = 0
        models_total = 0
        agreement_ratio = 0.0

    consensus = ConsensusSummary(
        majority_prediction=majority_pred,
        majority_label=CLASS_LABELS.get(majority_pred, f"Class {majority_pred}"),
        agreement_ratio=agreement_ratio,
        models_agreeing=models_agreeing,
        models_total=models_total
    )

    # 4. Echo subset of input features for UI confirmation
    input_echo = {
        "Administrative": request.features.Administrative,
        "ProductRelated": request.features.ProductRelated,
        "PageValues": request.features.PageValues,
        "Month": request.features.Month,
        "VisitorType": request.features.VisitorType,
        "Weekend": request.features.Weekend
    }

    return PredictionResponse(
        predictions=single_predictions,
        consensus=consensus,
        input_echo=input_echo
    )
```

### Expected Output & How to Verify It
Verify pipeline output via Python CLI:

```bash
python -c "from backend.services.model_service import model_container; model_container.load_artifacts(); from backend.schemas.predict import PredictionRequest, FeatureInput; from backend.services.model_service import run_prediction_pipeline; res = run_prediction_pipeline(PredictionRequest(features=FeatureInput())); print(res.model_dump_json(indent=2))"
```

**Expected terminal output snippet:**
```json
{
  "predictions": [
    {
      "model_name": "Logistic Regression",
      "model_key": "logistic_regression",
      "prediction": 0,
      "prediction_label": "No Purchase",
      "confidence": 0.8542,
      "inference_time_ms": 0.4521
    }
  ],
  "consensus": {
    "majority_prediction": 0,
    "majority_label": "No Purchase",
    "agreement_ratio": 1.0,
    "models_agreeing": 4,
    "models_total": 4
  }
}
```

### Common Pitfalls
1. **Unconverted NumPy Types:** `model.predict()` returns `numpy.int64` values. Returning raw `numpy.int64` or `numpy.float64` inside Pydantic responses causes JSON serialization errors unless cast explicitly with `int()` or `float()`.

### Git Checkpoint
- **Branch:** `backend/feature-predict-endpoint`
- **Commit:** `feat(predict-endpoint): implement multi-model prediction pipeline and consensus service`

---

## Step 3.3: HTTP Router Handler for POST /predict

### What & Why
Wiring the prediction service into a FastAPI router exposes `POST /api/v1/predict` with full Pydantic validation and Swagger documentation support.

### Code
Write `backend/routers/predict.py`:

```python
from fastapi import APIRouter, HTTPException, status
from backend.schemas.predict import PredictionRequest, PredictionResponse
from backend.services.model_service import run_prediction_pipeline, model_container

router = APIRouter(tags=["Prediction"])

@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_shopper_intention(request: PredictionRequest):
    """
    Submits e-commerce session features to all 5 pre-trained models.
    
    Returns side-by-side predictions, confidence scores, inference latencies,
    and consensus summary.
    """
    if not model_container.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models are not loaded into server memory."
        )

    try:
        response = run_prediction_pipeline(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution error: {str(e)}"
        )
```

Include router in `backend/main.py`:

```python
from backend.routers import health, predict

# ... setup app ...
app.include_router(health.router, prefix=settings.API_V1_STR)
app.include_router(predict.router, prefix=settings.API_V1_STR)
```

### Expected Output & How to Verify It
Start server and send a prediction POST request via `curl`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Administrative": 2,
      "Administrative_Duration": 50.0,
      "Informational": 0,
      "Informational_Duration": 0.0,
      "ProductRelated": 15,
      "ProductRelated_Duration": 350.5,
      "BounceRates": 0.0,
      "ExitRates": 0.01,
      "PageValues": 25.4,
      "SpecialDay": 0.0,
      "Month": "Nov",
      "OperatingSystems": 2,
      "Browser": 2,
      "Region": 1,
      "TrafficType": 2,
      "VisitorType": "Returning_Visitor",
      "Weekend": 0
    }
  }'
```

**Expected JSON Response (HTTP 200 OK):**
```json
{
  "predictions": [
    {
      "model_name": "Logistic Regression",
      "model_key": "logistic_regression",
      "prediction": 1,
      "prediction_label": "Purchase Completed",
      "confidence": 0.7842,
      "inference_time_ms": 0.512
    },
    {
      "model_name": "K-Nearest Neighbors",
      "model_key": "knn",
      "prediction": 1,
      "prediction_label": "Purchase Completed",
      "confidence": 0.8,
      "inference_time_ms": 2.145
    },
    {
      "model_name": "Support Vector Machine",
      "model_key": "svm",
      "prediction": 1,
      "prediction_label": "Purchase Completed",
      "confidence": 0.7412,
      "inference_time_ms": 0.812
    },
    {
      "model_name": "Naive Bayes",
      "model_key": "naive_bayes",
      "prediction": 1,
      "prediction_label": "Purchase Completed",
      "confidence": 0.6521,
      "inference_time_ms": 0.481
    }
  ],
  "consensus": {
    "majority_prediction": 1,
    "majority_label": "Purchase Completed",
    "agreement_ratio": 1.0,
    "models_agreeing": 4,
    "models_total": 4
  },
  "input_echo": {
    "Administrative": 2,
    "ProductRelated": 15,
    "PageValues": 25.4,
    "Month": "Nov",
    "VisitorType": "Returning_Visitor",
    "Weekend": 0
  }
}
```

### Common Pitfalls
1. **Missing Content-Type Header:** Omitting `-H "Content-Type: application/json"` in `curl` triggers a `422 Unprocessable Entity` or `400 Bad Request` from FastAPI.

### Git Checkpoint
- **Branch:** `backend/feature-predict-endpoint`
- **Commit:** `feat(predict-endpoint): register POST /api/v1/predict router handler`
