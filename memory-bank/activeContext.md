# Active Context

## Current Focus

Milestone 2 wrap-up and Milestone 3 preparation. The ML training pipeline is largely complete: all transformers (scaler, encoder) are serialized, 4 of 5 model `.pkl` files exist in `model_training/artifacts/` (logistic_regression, knn, svm, naive_bayes), and `metrics.json` + `feature_config.json` are present. **One gap: `decision_tree.pkl` is missing** — the git log claims "Implemented all ML algorithms" but the Decision Tree artifact was never serialized to disk (or was removed). This needs to be resolved before M2 can be marked fully complete.

The backend implementation guide (`/docs/backend-guide/`) has not been generated yet — it needs to be created before M3 work begins. The `/backend/` directory does not exist yet; no backend code has been written.

## What Exists

- `PRD.md` — full specification (v1.0.0, 1683 lines)
- `memory-bank/` — all 4 files (activeContext, progress, decisions, projectbrief)
- `model_training/artifacts/` — populated:
  - `scaler.pkl`, `encoder.pkl`, `feature_config.json` (M1 output)
  - `logistic_regression.pkl`, `knn.pkl`, `svm.pkl`, `naive_bayes.pkl` (4 of 5 models)
  - `metrics.json` (evaluation metrics for all models)
  - ❌ `decision_tree.pkl` — MISSING
- `docs/ml-guide/` — 10 guide files (00–09): workflow map, data understanding, preprocessing, visualization, per-model guides (LR, KNN, SVM, DT, NB), model comparison & serialization
- `docs/backend-guide/` — **does NOT exist yet** (needs to be generated)
- `backend/` — **does NOT exist** (M3 not started)
- Git: branches `backend`, `frontend`, `main`, `ml` exist. Recent commits show ML work merged via PR from `dev-ayush-17/ml`.

## What Does NOT Exist Yet

- `decision_tree.pkl` in `model_training/artifacts/`
- `backend/` directory and all backend code
- `docs/backend-guide/` directory and backend implementation guide
- `frontend/` directory (M4 not started)

## Immediate Next Steps

1. **Resolve missing `decision_tree.pkl`:** Re-run Decision Tree training and serialization, or confirm whether the existing artifacts are sufficient (check if `metrics.json` includes decision_tree metrics — if so, the model may just need to be re-serialized).
2. **Generate `/docs/backend-guide/`** — backend implementation guide following the pattern of `/docs/ml-guide/` (index + per-topic guides: project setup, model loading, predict endpoint, metrics endpoint, form-schema endpoint, health endpoint, input validation, CORS).
3. **Begin M3 — Backend implementation:** Create `backend/` directory, start with `main.py` (FastAPI app, startup event to load artifacts), then routers, schemas, services, and utils per PRD Section 6 folder structure. Work on branch `backend/feature-project-setup`.

## Blockers

- **Missing `decision_tree.pkl`:** M2 cannot be marked fully complete until all 5 model artifacts are on disk. Need to either re-serialize the Decision Tree or confirm the existing 4 models + metrics.json is acceptable.
- **No backend-guide docs:** M3 work should follow a written guide; the guide needs to be created first (or M3 can begin with the guide being written in parallel).

## Session Notes

- **2026-09-21:** Inspected actual disk state. Found M1 complete and M2 nearly complete (4/5 models + all artifacts). Memory bank was stale — claimed "pre-implementation, no code exists." Git log reveals ML work was contributed by `dev-ayush-17` via PR #1, merged into `main`. Decision Tree `.pkl` is missing from artifacts despite commit message claiming all algorithms implemented. Memory bank updated to reflect real state.
