# Backend Implementation Guide — 01: Project Setup & Structure

This document guides you through setting up the virtual environment, directory layout, dependency management, and minimal FastAPI server for the **Algorithm Arena** backend service.

---

## Step 1.1: Virtual Environment & Dependency Configuration

### What & Why
Setting up a dedicated virtual environment ensures isolated management of backend packages. ML model unpickling requires exact matching versions of `scikit-learn`, `numpy`, `pandas`, and `joblib` that were used during training in Milestone 2. Installing only web frameworks without the matching ML libraries will result in `ModuleNotFoundError` or unpickling `AttributeError` when loading saved `.pkl` artifacts.

### Code
Create the `requirements.txt` file at the repository root or inside `/backend/requirements.txt`:

```text
# Web Framework & Server
fastapi>=0.100.0,<1.0.0
uvicorn[standard]>=0.22.0,<1.0.0
pydantic>=2.0.0,<3.0.0

# Machine Learning Runtime (Must match training environment)
scikit-learn>=1.3.0,<1.6.0
pandas>=2.0.0,<3.0.0
numpy>=1.24.0,<2.0.0
joblib>=1.3.0,<2.0.0

# Utilities & Testing
httpx>=0.24.0
pytest>=7.3.0
python-dotenv>=1.0.0
```

To set up and activate the virtual environment from the repository root:

```bash
# Create virtual environment inside backend directory
python -m venv backend/venv

# Activate on Windows (PowerShell)
.\backend\venv\Scripts\Activate.ps1

# Activate on Linux/macOS
# source backend/venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

### Expected Output & How to Verify It
Verify that all packages are correctly installed by running a python one-liner to unpickle basic sklearn metadata:

```bash
python -c "import sklearn, pandas, numpy, joblib, fastapi, pydantic; print('All core libraries loaded successfully! sklearn version:', sklearn.__version__)"
```

**Expected terminal output:**
```text
All core libraries loaded successfully! sklearn version: 1.5.2
```

### Common Pitfalls
1. **Unpickling Version Mismatches:** Installing a newer major version of `scikit-learn` (e.g. 1.6 vs 1.3) can cause unpickling warnings or failures due to internal structure changes in `StandardScaler` or `OneHotEncoder`.
2. **Missing ML Libraries in Web Environment:** Omitting `pandas` or `scikit-learn` under the assumption that "the API only serves JSON" will break artifact deserialization, as `joblib.load()` requires class definitions present in the Python environment.

### Git Checkpoint
- **Branch:** `backend/feature-project-setup`
- **Commit:** `feat(setup): configure virtual environment and pin ML runtime dependencies`

---

## Step 1.2: Backend Directory Architecture

### What & Why
Organizing the backend codebase into clear separation of concerns (routers, schemas, services, config, utils) prevents tight coupling between HTTP routing logic and ML inference procedures.

### Code
Create the following directory layout inside `/backend`:

```text
AlgoArena/
└── backend/
    ├── __init__.py
    ├── main.py                     # FastAPI application entrypoint & lifespan
    ├── config.py                   # App configuration & artifact path settings
    ├── routers/                    # Endpoint route handlers
    │   ├── __init__.py
    │   ├── predict.py              # POST /api/v1/predict
    │   ├── metrics.py              # GET /api/v1/metrics
    │   ├── schema.py               # GET /api/v1/form-schema
    │   └── health.py               # GET /api/v1/health
    ├── schemas/                    # Pydantic request & response models
    │   ├── __init__.py
    │   ├── predict.py
    │   └── metrics.py
    ├── services/                   # Core business & ML inference logic
    │   ├── __init__.py
    │   ├── model_service.py        # Model loading & inference execution
    │   └── preprocessing_service.py # Feature transformation pipeline
    └── utils/                      # Timing & helper utilities
        ├── __init__.py
        └── timing.py
```

Powershell script to generate the folder structure:

```powershell
New-Item -ItemType Directory -Force -Path backend/routers, backend/schemas, backend/services, backend/utils
New-Item -ItemType File -Force -Path backend/__init__.py, backend/main.py, backend/config.py
New-Item -ItemType File -Force -Path backend/routers/__init__.py, backend/routers/predict.py, backend/routers/metrics.py, backend/routers/schema.py, backend/routers/health.py
New-Item -ItemType File -Force -Path backend/schemas/__init__.py, backend/schemas/predict.py, backend/schemas/metrics.py
New-Item -ItemType File -Force -Path backend/services/__init__.py, backend/services/model_service.py, backend/services/preprocessing_service.py
New-Item -ItemType File -Force -Path backend/utils/__init__.py, backend/utils/timing.py
```

### Expected Output & How to Verify It
Verify directory structure creation:

```powershell
Get-ChildItem -Recurse backend/
```

### Common Pitfalls
1. **Missing `__init__.py` Files:** Leaving folders without `__init__.py` files can lead to relative import failures when running tests or running via `uvicorn backend.main:app`.

### Git Checkpoint
- **Branch:** `backend/feature-project-setup`
- **Commit:** `feat(setup): create modular directory structure for routers, schemas, services, and utils`

---

## Step 1.3: Central Configuration Module

### What & Why
`config.py` centralizes paths to serialized model artifacts (`/model_training/artifacts/`) and application settings. Hardcoding absolute paths across router files creates fragile code that breaks on deployment or across different developer machines.

### Code
Write `backend/config.py`:

```python
import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Information
    PROJECT_NAME: str = "Algorithm Arena API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Path Resolution
    BACKEND_DIR: Path = Path(__file__).resolve().parent
    PROJECT_ROOT: Path = BACKEND_DIR.parent
    ARTIFACTS_DIR: Path = PROJECT_ROOT / "model_training" / "artifacts"
    
    # Artifact Filenames
    SCALER_FILENAME: str = "scaler.pkl"
    ENCODER_FILENAME: str = "encoder.pkl"
    FEATURE_CONFIG_FILENAME: str = "feature_config.json"
    METRICS_FILENAME: str = "metrics.json"
    
    # Model Keys & Filenames
    MODEL_FILES: dict[str, str] = {
        "logistic_regression": "logistic_regression.pkl",
        "knn": "knn.pkl",
        "svm": "svm.pkl",
        "decision_tree": "decision_tree.pkl",
        "naive_bayes": "naive_bayes.pkl"
    }

    class Config:
        case_sensitive = True

settings = Settings()
```

*Note: If `pydantic-settings` is not installed separately, standard Python `os.getenv` or basic dataclass / class config can be used.*

Alternative lightweight `backend/config.py`:

```python
import os
from pathlib import Path

class Settings:
    PROJECT_NAME: str = "Algorithm Arena API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    BACKEND_DIR: Path = Path(__file__).resolve().parent
    PROJECT_ROOT: Path = BACKEND_DIR.parent
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

### Expected Output & How to Verify It
Verify path resolution using Python CLI:

```bash
python -c "from backend.config import settings; print('Artifacts dir exists:', settings.ARTIFACTS_DIR.exists(), '| Path:', settings.ARTIFACTS_DIR)"
```

**Expected terminal output:**
```text
Artifacts dir exists: True | Path: C:\Users\...\AlgoArena\model_training\artifacts
```

### Common Pitfalls
1. **Relative Working Directory Errors:** Using `Path("model_training/artifacts")` instead of resolving relative to `__file__` causes file-not-found errors depending on where `uvicorn` is launched from.

### Git Checkpoint
- **Branch:** `backend/feature-project-setup`
- **Commit:** `feat(setup): add configuration settings for artifact resolution`

---

## Step 1.4: Minimal Health Server ("Hello World")

### What & Why
Before wiring complex ML loading code, a bare-bones FastAPI application verifies that Uvicorn, route inclusion, and CORS middleware work properly.

### Code
Write `backend/routers/health.py`:

```python
from fastapi import APIRouter
from backend.config import settings

router = APIRouter(tags=["Health"])

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION
    }
```

Write `backend/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.routers import health

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(health.router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {"message": "Welcome to Algorithm Arena API. Visit /docs for OpenAPI specifications."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
```

### Expected Output & How to Verify It
Start the server:

```bash
uvicorn backend.main:app --reload --port 8000
```

In a second terminal, execute curl:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

**Expected JSON response:**
```json
{
  "status": "healthy",
  "project": "Algorithm Arena API",
  "version": "1.0.0"
}
```

### Common Pitfalls
1. **Wrong Import Paths:** Running `uvicorn main:app` from inside `/backend` vs `uvicorn backend.main:app` from project root can cause `ModuleNotFoundError: No module named 'backend'`. Always set `PYTHONPATH=.` or run from repository root.

### Git Checkpoint
- **Branch:** `backend/feature-project-setup`
- **Commit:** `feat(setup): implement minimal health server and fastapi boilerplate`
