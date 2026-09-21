# Backend Implementation Guide — 04: Metrics & Form-Schema Endpoints

This document details implementing the benchmark evaluation endpoint (`GET /api/v1/metrics`) and the dynamic UI metadata endpoint (`GET /api/v1/form-schema`).

---

## Step 4.1: Static Benchmark Metrics Endpoint (`GET /metrics`)

### What & Why
The Report Card UI compares accuracy, precision, recall, F1 score, average inference time, and serialized file size across all 5 models. 

> **PERFORMANCE RULE:** The backend MUST read pre-computed benchmark metrics from `model_training/artifacts/metrics.json` on disk or from startup memory cache. It MUST NEVER recalculate evaluation metrics on live test data during an HTTP request.

### Code
Write `backend/schemas/metrics.py`:

```python
from pydantic import BaseModel
from typing import Dict, List, Any, Optional

class ModelMetricDetail(BaseModel):
    display_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_inference_time_ms: float
    model_file_size_kb: float
    training_time_seconds: float

class DatasetSizeInfo(BaseModel):
    total_rows: int
    train_rows: int
    test_rows: int
    features: int
    features_after_encoding: int

class MetricsResponse(BaseModel):
    generated_at: str
    dataset: str
    dataset_size: DatasetSizeInfo
    models: Dict[str, ModelMetricDetail]
    best_per_metric: Optional[Dict[str, str]] = None
```

Write `backend/routers/metrics.py`:

```python
import json
import logging
from fastapi import APIRouter, HTTPException, status
from backend.config import settings
from backend.schemas.metrics import MetricsResponse

router = APIRouter(tags=["Metrics"])
logger = logging.getLogger("backend.routers.metrics")

# In-memory cache for metrics payload
_metrics_cache: Optional[dict] = None

def load_metrics_json() -> dict:
    global _metrics_cache
    if _metrics_cache is not None:
        return _metrics_cache

    metrics_path = settings.ARTIFACTS_DIR / settings.METRICS_FILENAME
    if not metrics_path.exists():
        logger.error(f"metrics.json missing at {metrics_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Benchmark metrics artifact missing."
        )

    try:
        with open(metrics_path, "r") as f:
            data = json.load(f)
            
        # Compute best model per metric for Report Card highlighting
        models = data.get("models", {})
        if models:
            best_per_metric = {
                "highest_accuracy": max(models.items(), key=lambda x: x[1].get("accuracy", 0))[0],
                "highest_f1": max(models.items(), key=lambda x: x[1].get("f1_score", 0))[0],
                "fastest_inference": min(models.items(), key=lambda x: x[1].get("avg_inference_time_ms", float("inf")))[0],
                "smallest_file": min(models.items(), key=lambda x: x[1].get("model_file_size_kb", float("inf")))[0]
            }
            data["best_per_metric"] = best_per_metric

        _metrics_cache = data
        return data
    except Exception as e:
        logger.error(f"Error reading metrics.json: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read benchmark metrics: {str(e)}"
        )

@router.get("/metrics", response_model=MetricsResponse, status_code=status.HTTP_200_OK)
def get_model_benchmark_metrics():
    """
    Retrieves pre-computed evaluation metrics (Accuracy, Precision, Recall, F1, 
    Inference Latency, File Size) for all trained models.
    """
    return load_metrics_json()
```

### Expected Output & How to Verify It
Start server and query endpoint:

```bash
curl http://127.0.0.1:8000/api/v1/metrics
```

**Expected JSON Response (HTTP 200 OK):**
```json
{
  "generated_at": "2026-09-20T20:18:03.788924+00:00",
  "dataset": "Online Shoppers Purchasing Intention Dataset (UCI ID 468)",
  "dataset_size": {
    "total_rows": 12330,
    "train_rows": 9864,
    "test_rows": 2466,
    "features": 17,
    "features_after_encoding": 28
  },
  "models": {
    "logistic_regression": {
      "display_name": "Logistic Regression",
      "accuracy": 0.85,
      "precision": 0.5107,
      "recall": 0.7487,
      "f1_score": 0.6072,
      "avg_inference_time_ms": 0.4587,
      "model_file_size_kb": 1.73,
      "training_time_seconds": 0.0459
    },
    "knn": {
      "display_name": "K-Nearest Neighbors",
      "accuracy": 0.8719,
      "precision": 0.6528,
      "recall": 0.3691,
      "f1_score": 0.4716,
      "avg_inference_time_ms": 1.7593,
      "model_file_size_kb": 2236.27,
      "training_time_seconds": 0.0029
    }
  },
  "best_per_metric": {
    "highest_accuracy": "knn",
    "highest_f1": "svm",
    "fastest_inference": "logistic_regression",
    "smallest_file": "logistic_regression"
  }
}
```

### Common Pitfalls
1. **Re-reading File per Request:** Opening `metrics.json` on disk for every HTTP request adds unnecessary disk I/O. Caching the parsed dictionary in memory (`_metrics_cache`) speeds up execution.

### Git Checkpoint
- **Branch:** `backend/feature-metrics-endpoint`
- **Commit:** `feat(metrics-endpoint): implement GET /api/v1/metrics with in-memory caching`

---

## Step 4.2: Dynamic Form Schema Endpoint (`GET /form-schema`)

### What & Why
Hardcoding feature input fields, dropdown options, and validation bounds into frontend HTML creates tight coupling. `GET /api/v1/form-schema` exposes dataset metadata from `model_training/artifacts/feature_config.json` so the frontend can dynamically build form inputs.

### Code
Write `backend/routers/schema.py`:

```python
import json
import logging
from fastapi import APIRouter, HTTPException, status
from backend.config import settings

router = APIRouter(tags=["Schema"])
logger = logging.getLogger("backend.routers.schema")

_schema_cache: dict = None

def get_fallback_feature_config() -> dict:
    """Returns fallback schema if feature_config.json is missing or invalid."""
    return {
        "numerical_features": [
            {"name": "Administrative", "dtype": "int64", "min": 0, "max": 27, "default": 0, "description": "Number of administrative pages visited"},
            {"name": "Administrative_Duration", "dtype": "float64", "min": 0.0, "max": 3398.0, "default": 0.0, "description": "Total time spent on administrative pages (seconds)"},
            {"name": "Informational", "dtype": "int64", "min": 0, "max": 24, "default": 0, "description": "Number of informational pages visited"},
            {"name": "Informational_Duration", "dtype": "float64", "min": 0.0, "max": 2549.0, "default": 0.0, "description": "Total time spent on informational pages (seconds)"},
            {"name": "ProductRelated", "dtype": "int64", "min": 0, "max": 705, "default": 1, "description": "Number of product-related pages visited"},
            {"name": "ProductRelated_Duration", "dtype": "float64", "min": 0.0, "max": 63973.5, "default": 10.0, "description": "Total time spent on product-related pages (seconds)"},
            {"name": "BounceRates", "dtype": "float64", "min": 0.0, "max": 0.2, "default": 0.0, "description": "Percentage of visitors entering site and leaving immediately"},
            {"name": "ExitRates", "dtype": "float64", "min": 0.0, "max": 0.2, "default": 0.02, "description": "Percentage of pageviews that were last in session"},
            {"name": "PageValues", "dtype": "float64", "min": 0.0, "max": 361.0, "default": 0.0, "description": "Average Google Analytics value of visited pages"},
            {"name": "SpecialDay", "dtype": "float64", "min": 0.0, "max": 1.0, "default": 0.0, "description": "Closeness of site visit time to a special day (0.0 to 1.0)"}
        ],
        "categorical_features": [
            {"name": "Month", "dtype": "string", "categories": ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], "default": "May", "description": "Month of the session"},
            {"name": "OperatingSystems", "dtype": "int64", "categories": [1, 2, 3, 4, 5, 6, 7, 8], "default": 1, "description": "Operating system ID"},
            {"name": "Browser", "dtype": "int64", "categories": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "default": 2, "description": "Browser ID"},
            {"name": "Region", "dtype": "int64", "categories": [1, 2, 3, 4, 5, 6, 7, 8, 9], "default": 1, "description": "Geographic region ID"},
            {"name": "TrafficType", "dtype": "int64", "categories": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "default": 2, "description": "Traffic source type ID"},
            {"name": "VisitorType", "dtype": "string", "categories": ["Returning_Visitor", "New_Visitor", "Other"], "default": "Returning_Visitor", "description": "Visitor status"},
            {"name": "Weekend", "dtype": "int64", "categories": [0, 1], "default": 0, "description": "Session occurred on weekend (0=No, 1=Yes)"}
        ],
        "target": {
            "name": "Revenue",
            "classes": [0, 1],
            "class_labels": ["No Purchase", "Purchase Completed"]
        }
    }

@router.get("/form-schema", status_code=status.HTTP_200_OK)
def get_form_schema():
    """
    Returns feature definitions, valid ranges, categories, and defaults
    for dynamic frontend form rendering.
    """
    global _schema_cache
    if _schema_cache is not None:
        return _schema_cache

    config_path = settings.ARTIFACTS_DIR / settings.FEATURE_CONFIG_FILENAME
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                content = json.load(f)
                if isinstance(content, dict) and "numerical_features" in content:
                    _schema_cache = content
                    return _schema_cache
        except Exception as e:
            logger.warning(f"Could not parse {config_path}: {e}")

    # Fallback to hardcoded valid schema definition
    _schema_cache = get_fallback_feature_config()
    return _schema_cache
```

Include routers in `backend/main.py`:

```python
from backend.routers import metrics, schema

app.include_router(metrics.router, prefix=settings.API_V1_STR)
app.include_router(schema.router, prefix=settings.API_V1_STR)
```

### Expected Output & How to Verify It
Start server and curl form-schema endpoint:

```bash
curl http://127.0.0.1:8000/api/v1/form-schema
```

**Expected JSON response snippet:**
```json
{
  "numerical_features": [
    {
      "name": "Administrative",
      "dtype": "int64",
      "min": 0,
      "max": 27,
      "default": 0
    }
  ],
  "categorical_features": [
    {
      "name": "Month",
      "dtype": "string",
      "categories": [
        "Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
      ]
    }
  ]
}
```

### Common Pitfalls
1. **Unchecked Invalid `feature_config.json`:** If `feature_config.json` was saved with path string instead of dict (as noted in artifact inspection), the route could return a string instead of JSON. The fallback logic in `get_fallback_feature_config()` guarantees valid JSON delivery.

### Git Checkpoint
- **Branch:** `backend/feature-metrics-endpoint`
- **Commit:** `feat(metrics-endpoint): implement GET /api/v1/form-schema with fallback metadata`
