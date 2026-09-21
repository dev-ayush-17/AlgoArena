# Backend Implementation Guide — 06: Testing the API

This document guides you through creating a comprehensive automated test suite for the FastAPI backend using `pytest` and `httpx` (`TestClient`). The test suite validates happy-path endpoints, Pydantic validation rules, ML inference latency measurement, and edge cases.

---

## Step 6.1: Pytest Test Suite Configuration & Fixtures

### What & Why
Setting up reusable `pytest` fixtures initializes the FastAPI `TestClient` and verifies that model artifacts are pre-loaded in memory before running test assertions.

### Code
Create test directory layout inside `/backend`:

```text
backend/
└── tests/
    ├── __init__.py
    ├── conftest.py               # TestClient fixtures and startup hooks
    ├── test_health.py            # GET /health assertions
    ├── test_metrics.py           # GET /metrics and GET /form-schema assertions
    ├── test_predict.py           # POST /predict happy paths & latency checks
    └── test_edge_cases.py        # Validation errors (422, 400) & partial failures
```

Write `backend/tests/conftest.py`:

```python
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.model_service import model_container

@pytest.fixture(scope="session", autouse=True)
def initialize_artifacts():
    """Ensure artifacts are loaded once before running test session."""
    if not model_container.is_loaded:
        model_container.load_artifacts()

@pytest.fixture(scope="module")
def client():
    """Provides a reusable FastAPI TestClient instance."""
    with TestClient(app) as c:
        yield c
```

### Expected Output & How to Verify It
Run `pytest` CLI from backend directory:

```bash
pytest backend/tests -v
```

**Expected output:**
```text
collected 0 items
```

### Common Pitfalls
1. **Uninitialized Lifespan in Tests:** `TestClient(app)` using standard context manager automatically triggers the lifespan context (`lifespan(app)`). Omitting `with TestClient(app) as c:` can prevent lifespan startup from firing in older FastAPI versions.

### Git Checkpoint
- **Branch:** `backend/feature-testing`
- **Commit:** `feat(testing): configure pytest test suite setup and TestClient fixtures`

---

## Step 6.2: Happy Path Tests (`/health`, `/metrics`, `/form-schema`)

### What & Why
Happy-path tests verify that all GET endpoints return HTTP 200 status codes and valid JSON structures matching schema contracts.

### Code
Write `backend/tests/test_health.py`:

```python
def test_health_check_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] > 0
    assert "logistic_regression" in data["models"]
    assert data["transformers_loaded"] is True
```

Write `backend/tests/test_metrics.py`:

```python
def test_get_metrics_endpoint(client):
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "dataset" in data
    assert "models" in data
    assert "logistic_regression" in data["models"]
    
    # Assert specific metric keys
    lr = data["models"]["logistic_regression"]
    assert 0.0 <= lr["accuracy"] <= 1.0
    assert 0.0 <= lr["f1_score"] <= 1.0
    assert lr["avg_inference_time_ms"] >= 0.0
    assert "best_per_metric" in data

def test_get_form_schema_endpoint(client):
    response = client.get("/api/v1/form-schema")
    assert response.status_code == 200
    data = response.json()
    assert "numerical_features" in data
    assert "categorical_features" in data
    assert len(data["numerical_features"]) == 10
    assert len(data["categorical_features"]) == 7
```

### Expected Output & How to Verify It
Run pytest:

```bash
pytest backend/tests/test_health.py backend/tests/test_metrics.py -v
```

**Expected output:**
```text
backend/tests/test_health.py::test_health_check_endpoint PASSED             [ 33%]
backend/tests/test_metrics.py::test_get_metrics_endpoint PASSED              [ 66%]
backend/tests/test_metrics.py::test_get_form_schema_endpoint PASSED         [100%]
```

### Common Pitfalls
1. **Hardcoding Exact Accuracy Values:** Asserting `assert lr["accuracy"] == 0.85` will fail if models are re-trained on a slightly updated dataset split. Asserting range bounds (`0.0 <= accuracy <= 1.0`) is robust.

### Git Checkpoint
- **Branch:** `backend/feature-testing`
- **Commit:** `feat(testing): add happy path tests for health, metrics, and form-schema endpoints`

---

## Step 6.3: Prediction & Latency Benchmark Tests (`POST /predict`)

### What & Why
Tests `POST /predict` with realistic feature payloads from the Online Shoppers dataset, asserting that 4 model predictions are returned, probabilities sum correctly, and inference timing is present.

### Code
Write `backend/tests/test_predict.py`:

```python
import pytest

@pytest.fixture
def valid_shopper_payload():
    return {
        "features": {
            "Administrative": 3,
            "Administrative_Duration": 87.0,
            "Informational": 1,
            "Informational_Duration": 12.0,
            "ProductRelated": 22,
            "ProductRelated_Duration": 640.5,
            "BounceRates": 0.0,
            "ExitRates": 0.015,
            "PageValues": 18.5,
            "SpecialDay": 0.0,
            "Month": "Nov",
            "OperatingSystems": 2,
            "Browser": 2,
            "Region": 1,
            "TrafficType": 2,
            "VisitorType": "Returning_Visitor",
            "Weekend": 0
        }
    }

def test_predict_endpoint_happy_path(client, valid_shopper_payload):
    response = client.post("/api/v1/predict", json=valid_shopper_payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "predictions" in data
    assert "consensus" in data
    assert "input_echo" in data
    
    # Assert model list structure
    predictions = data["predictions"]
    assert len(predictions) >= 4
    
    for pred in predictions:
        assert pred["model_key"] in ["logistic_regression", "knn", "svm", "naive_bayes", "decision_tree"]
        assert pred["prediction"] in [0, 1]
        assert pred["prediction_label"] in ["No Purchase", "Purchase Completed"]
        assert 0.0 <= pred["confidence"] <= 1.0
        # Latency check: prediction should take under 500ms
        assert 0.0 <= pred["inference_time_ms"] < 500.0

    # Consensus checks
    consensus = data["consensus"]
    assert consensus["majority_prediction"] in [0, 1]
    assert 0.0 <= consensus["agreement_ratio"] <= 1.0
```

### Expected Output & How to Verify It
Run test file:

```bash
pytest backend/tests/test_predict.py -v
```

**Expected output:**
```text
backend/tests/test_predict.py::test_predict_endpoint_happy_path PASSED      [100%]
```

### Common Pitfalls
1. **Ignoring Microsecond Timing Assertions:** Omitting assertions on `inference_time_ms` allows regression bugs (such as un-cached model loading or disk I/O on hot paths) to pass silently.

### Git Checkpoint
- **Branch:** `backend/feature-testing`
- **Commit:** `feat(testing): add prediction endpoint happy-path and latency assertions`

---

## Step 6.4: Edge Case & Validation Error Suite

### What & Why
Automated tests MUST explicitly attempt to break the API by passing missing fields, out-of-range numericals, and invalid categorical strings to confirm that `422 Unprocessable Entity` and `400 Bad Request` are returned cleanly without unhandled 500 server crashes.

### Code
Write `backend/tests/test_edge_cases.py`:

```python
def test_missing_features_payload_returns_422(client):
    # Missing required 'features' object
    response = client.post("/api/v1/predict", json={})
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "validation_error"

def test_invalid_type_field_returns_422(client):
    # Passing string for integer feature 'Administrative'
    payload = {
        "features": {
            "Administrative": "not_a_number"
        }
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "validation_error"

def test_out_of_range_bounce_rates_returns_400(client, valid_shopper_payload):
    # BounceRates > 1.0 violates physical ratio bounds
    payload = valid_shopper_payload.copy()
    payload["features"]["BounceRates"] = 5.0
    
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "invalid_feature_value"
    assert "BounceRates" in data["message"]

def test_invalid_month_category_returns_422(client, valid_shopper_payload):
    # Invalid Month category 'December' (dataset expects 'Dec')
    payload = valid_shopper_payload.copy()
    payload["features"]["Month"] = "December"
    
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "validation_error"
```

### Expected Output & How to Verify It
Run complete test suite across all files:

```bash
pytest backend/tests -v
```

**Expected output:**
```text
backend/tests/test_edge_cases.py::test_missing_features_payload_returns_422 PASSED        [ 25%]
backend/tests/test_edge_cases.py::test_invalid_type_field_returns_422 PASSED              [ 50%]
backend/tests/test_edge_cases.py::test_out_of_range_bounce_rates_returns_400 PASSED       [ 75%]
backend/tests/test_edge_cases.py::test_invalid_month_category_returns_422 PASSED          [100%]

============================== 9 passed in 1.42s ==============================
```

### Common Pitfalls
1. **Testing Only Happy Path:** Neglecting edge-case tests leaves the backend vulnerable to unhandled server crashes when deployed live to web clients.

### Git Checkpoint
- **Branch:** `backend/feature-testing`
- **Commit:** `feat(testing): add edge case suite for 422 validation and 400 boundary violations`
