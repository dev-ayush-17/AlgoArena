# Backend Implementation Guide — 00: Workflow & Git Map

This guide serves as the index and roadmap for building the **Algorithm Arena** backend service. The backend is a lightweight, high-performance REST API built with FastAPI that serves pre-trained machine learning models for real-time inference and benchmarks model metrics.

---

## 1. Branch Strategy & Git Workflow Map

All work is conducted on feature branches derived from the `backend` long-lived branch. Each feature branch corresponds to a specific guide file and is merged back into `backend` via a Pull Request (PR) after passing local verification and tests. Once all backend features are implemented and validated against `PRD.md`, `backend` is merged into `main`.

| Step | Guide File | Git Branch | Commit Scope & Convention | Description |
|---|---|---|---|---|
| **01** | `01-project-setup-and-structure.md` | `backend/feature-project-setup` | `feat(setup): ...` | Directory layout, dependencies (`scikit-learn`, `fastapi`, `pydantic`), venv, and health check endpoint. |
| **02** | `02-loading-artifacts-and-schemas.md` | `backend/feature-model-loading` | `feat(model-loading): ...` | Load models & fitted transformers into memory at startup; define request/response Pydantic schemas. |
| **03** | `03-predict-endpoint.md` | `backend/feature-predict-endpoint` | `feat(predict-endpoint): ...` | `POST /api/v1/predict` handler with preprocessing, 5-model execution, timing, and consensus logic. |
| **04** | `04-metrics-endpoint.md` | `backend/feature-metrics-endpoint` | `feat(metrics-endpoint): ...` | `GET /api/v1/metrics` and `GET /api/v1/form-schema` endpoints serving pre-computed benchmark metrics. |
| **05** | `05-error-handling-and-edge-cases.md` | `backend/feature-error-handling` | `feat(error-handling): ...` | Custom exception handlers for validation, out-of-range categoricals, model failure, and 422/400/500 shapes. |
| **06** | `06-testing-the-api.md` | `backend/feature-testing` | `feat(testing): ...` | `pytest` test suite with `TestClient` covering happy paths, edge cases, and inference latency validation. |
| **07** | `07-deployment-prep.md` | `backend/feature-deployment-config` | `feat(deploy-config): ...` | Environment configuration, CORS, artifact shipping strategy, and Render pre-deployment checklist. |

---

## 2. Commit Message Convention

Follow standard Conventional Commits for all commits on feature branches:

- `feat(setup): add fastapi app boilerplate and directory structure`
- `feat(model-loading): implement startup artifact loader and pydantic feature schemas`
- `feat(predict-endpoint): add POST /predict with preprocessing and model timing`
- `feat(metrics-endpoint): implement GET /metrics and GET /form-schema`
- `feat(error-handling): add custom handlers for ML input validation and model fallback`
- `feat(testing): add pytest suite for API endpoints and edge cases`
- `feat(deploy-config): configure CORS, uvicorn entrypoint, and deployment checklist`

---

## 3. Guide Table of Contents

1. **[01 — Project Setup & Structure](file:///docs/backend-guide/01-project-setup-and-structure.md)**
   - Virtual environment configuration, pinning ML runtime dependencies, setting up modular package layout (`routers`, `schemas`, `services`, `utils`), and basic FastAPI health server.
2. **[02 — Loading Artifacts & Schemas](file:///docs/backend-guide/02-loading-artifacts-and-schemas.md)**
   - FastAPI lifespan event management, singleton state for `scaler.pkl`, `encoder.pkl`, and 5 `.pkl` models, and strict feature input validation schemas matching Online Shoppers Purchasing Intention dataset.
3. **[03 — Predict Endpoint](file:///docs/backend-guide/03-predict-endpoint.md)**
   - Single-row transformation pipeline, high-precision latency measurement (`time.perf_counter`), side-by-side inference across all 5 models, and majority voting consensus calculation.
4. **[04 — Metrics & Form-Schema Endpoints](file:///docs/backend-guide/04-metrics-endpoint.md)**
   - Serving static benchmark data from `metrics.json` without runtime recomputation, and delivering dynamic form configuration for UI generation (`GET /api/v1/form-schema`).
5. **[05 — Error Handling & Edge Cases](file:///docs/backend-guide/05-error-handling-and-edge-cases.md)**
   - Handling out-of-range categoricals, unseen labels, missing numerical features, corrupt binary outputs, and partial failure responses when individual models fail.
6. **[06 — Testing the API](file:///docs/backend-guide/06-testing-the-api.md)**
   - Writing unit and integration tests using `pytest` and `httpx.AsyncClient` / `TestClient` to verify prediction stability, schema validation, and endpoint latencies.
7. **[07 — Deployment Prep](file:///docs/backend-guide/07-deployment-prep.md)**
   - Production Uvicorn deployment settings, environment variable management, CORS policies, artifact storage strategies for cloud platforms (e.g. Render), and release readiness checklist.
