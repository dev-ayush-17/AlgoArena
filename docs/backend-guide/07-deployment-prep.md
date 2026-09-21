# Backend Implementation Guide — 07: Deployment Preparation

This document prepares the FastAPI backend for public deployment (e.g. Render, Railway, or Fly.io), detailing environment variable management, CORS policies, model artifact shipping strategies, `.gitignore` rules, and a pre-deployment readiness checklist strictly aligned with **PRD.md Section 13 (Deployment Plan)**.

---

## Step 7.1: Environment Variable Management & Production Settings

### What & Why
Hardcoding localhost URLs, environment flags, or port numbers makes deployment fragile across cloud environments. Managing settings via environment variables using `python-dotenv` enables smooth transitions between local development and cloud hosting.

### Code
Create `.env.example` in project root:

```text
# Server Configuration
ENVIRONMENT=production
PORT=8000
LOG_LEVEL=info

# CORS Origins (Comma-separated list or wildcard)
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:8000,https://your-frontend-domain.onrender.com

# Artifact Storage Directory Path
ARTIFACTS_DIR=model_training/artifacts
```

Update `backend/config.py` to read environment variables:

```python
import os
from pathlib import Path

class Settings:
    PROJECT_NAME: str = "Algorithm Arena API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    PORT: int = int(os.getenv("PORT", 8000))
    
    # CORS Configuration
    ALLOWED_ORIGINS: list = os.getenv(
        "ALLOWED_ORIGINS", 
        "*"
    ).split(",")

    # Directory Paths
    BACKEND_DIR: Path = Path(__file__).resolve().parent
    PROJECT_ROOT: Path = BACKEND_DIR.parent
    
    env_artifacts = os.getenv("ARTIFACTS_DIR")
    if env_artifacts:
        ARTIFACTS_DIR: Path = Path(env_artifacts).resolve()
    else:
        ARTIFACTS_DIR: Path = PROJECT_ROOT / "model_training" / "artifacts"
    
    SCALER_FILENAME: str = "scaler.pkl"
    ENCODER_FILENAME: str = "encoder.pkl"
    FEATURE_CONFIG_FILENAME: str = "feature_config.json"
    METRICS_FILENAME: str = "metrics.json"
    
    MODEL_FILES: dict = {
        "logistic_regression": "logistic_regression.pkl",
        "knn": "knn.pkl",
        "svm": "svm.pkl",
        "decision_tree": "decision_tree.pkl",
        "naive_bayes": "naive_bayes.pkl"
    }

settings = Settings()
```

Update CORS in `backend/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Expected Output & How to Verify It
Verify settings resolution with custom env vars:

```bash
ENVIRONMENT=production PORT=9000 python -c "from backend.config import settings; print('Env:', settings.ENVIRONMENT, '| Port:', settings.PORT)"
```

**Expected output:**
```text
Env: production | Port: 9000
```

### Common Pitfalls
1. **Hardcoding Host/Port in Uvicorn:** Hardcoding `host="127.0.0.1"` in Uvicorn startup scripts on Render or Docker prevents external HTTP traffic from binding. Cloud platforms require binding to `host="0.0.0.0"` and reading `PORT` from environment variables.

### Git Checkpoint
- **Branch:** `backend/feature-deployment-config`
- **Commit:** `feat(deploy-config): configure environment variables and production CORS settings`

---

## Step 7.2: Git Shipping Strategy vs Gitignore Audit

### What & Why
Cloud deployment platforms (like Render free tier) have repository size constraints (100MB per file Git limit). Inspecting model artifact sizes:
- `knn.pkl`: ~2.2 MB
- `svm.pkl`: ~1.1 MB
- `logistic_regression.pkl`: ~2 KB
- `naive_bayes.pkl`: ~2 KB
- `scaler.pkl` & `encoder.pkl`: ~2 KB
- **Total artifacts directory size:** **~3.4 MB**

Because total serialized model artifacts are under 4 MB, **all model artifacts CAN and SHOULD be committed directly into Git inside `model_training/artifacts/`**. Large raw training datasets (e.g., `model_training/data/raw/*.csv`) must remain `.gitignored`.

### Code
Create/update `.gitignore` at repository root:

```text
# Python Environment & Cache
__pycache__/
*.py[cod]
*$py.class
venv/
.venv/
env/
.env

# Model Training Data (Keep raw CSV out of Git if >50MB)
# model_training/data/raw/*.csv

# System & IDE files
.DS_Store
.vscode/
.idea/

# DO NOT IGNORE MODEL ARTIFACTS
!model_training/artifacts/*.pkl
!model_training/artifacts/*.json
```

### Expected Output & How to Verify It
Check git status to ensure artifacts are tracked while raw data / venvs are ignored:

```bash
git status
```

**Expected output:**
Artifacts inside `model_training/artifacts/` are tracked; `venv/` and `.env` are ignored.

### Common Pitfalls
1. **Accidentally Gitignoring `.pkl` Files:** Beginners often put `*.pkl` into `.gitignore`. When deployed to Render, server startup fails immediately with `FileNotFoundError: scaler.pkl missing`.

### Git Checkpoint
- **Branch:** `backend/feature-deployment-config`
- **Commit:** `feat(deploy-config): update gitignore rules to ship ML model artifacts`

---

## Step 7.3: Render Web Service Start Command & Deployment Entrypoint

### What & Why
Configuring the deployment build and start commands for Render or Docker ensures seamless container boot.

### Code
Create `render.yaml` in repository root (optional blueprint definition):

```yaml
services:
  - type: web
    name: algoarena-backend
    env: python
    region: oregon
    buildCommand: pip install -r backend/requirements.txt
    startCommand: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: ENVIRONMENT
        value: production
      - key: ALLOWED_ORIGINS
        value: "*"
```

Direct Start Command for Render Dashboard:

- **Build Command:** `pip install -r backend/requirements.txt`
- **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

### Expected Output & How to Verify It
Test startup command locally using simulated PORT env variable:

```bash
PORT=8000 uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Curl local server bound to 0.0.0.0:

```bash
curl http://0.0.0.0:8000/api/v1/health
```

### Common Pitfalls
1. **Forgetting `--host 0.0.0.0`:** Binding to `127.0.0.1` causes Render health checks to fail with port timeout errors.

### Git Checkpoint
- **Branch:** `backend/feature-deployment-config`
- **Commit:** `feat(deploy-config): configure render deployment entrypoint and start command`

---

## Step 7.4: Pre-Deployment Verification Checklist

### What & Why
Before merging `backend` into `main` and deploying to Render, verify all operational criteria from **PRD.md Section 13**.

### Checklist

| # | Check Item | Status | Verification Command |
|---|---|---|---|
| 1 | **Artifact Integrity:** All 5 models + scaler + encoder exist in `/model_training/artifacts/`. | 🟢 Ready | `ls model_training/artifacts/*.pkl` |
| 2 | **Lifespan Startup:** FastAPI initializes artifact container on boot without errors. | 🟢 Ready | `curl http://127.0.0.1:8000/api/v1/health` |
| 3 | **Prediction Logic:** `POST /predict` returns side-by-side predictions for all models. | 🟢 Ready | `pytest backend/tests/test_predict.py` |
| 4 | **Microsecond Timing:** Latencies in `predictions[].inference_time_ms` are > 0.0ms. | 🟢 Ready | `pytest backend/tests/test_predict.py` |
| 5 | **Validation & Error Shapes:** `422`, `400`, and `500` return custom JSON error structures. | 🟢 Ready | `pytest backend/tests/test_edge_cases.py` |
| 6 | **Test Coverage:** All unit/integration tests pass with 0 errors. | 🟢 Ready | `pytest backend/tests -v` |
| 7 | **CORS Policy:** Frontend domain allowed in `ALLOWED_ORIGINS`. | 🟢 Ready | `curl -H "Origin: http://localhost:3000" -I http://127.0.0.1:8000/api/v1/health` |
| 8 | **Port Binding:** Start command binds to `0.0.0.0:$PORT`. | 🟢 Ready | `uvicorn backend.main:app --host 0.0.0.0 --port 8000` |

### Git Checkpoint
- **Branch:** `backend` (after PR merge from `backend/feature-deployment-config`)
- **Commit:** `feat(deploy-config): complete pre-deployment verification checklist for release`
