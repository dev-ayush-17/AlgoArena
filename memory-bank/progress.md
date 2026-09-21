# Progress Log

## Milestones

- ✅ **M1: Data Pipeline** — Complete
  - [x] 1.1 Download and inspect dataset — Online Shoppers Purchasing Intention (UCI ID 468) loaded
  - [x] 1.2 Implement preprocessing.py — scaler.pkl, encoder.pkl, feature_config.json produced
  - [x] 1.3 Verify no data leakage — transformers fitted on train set only
  - [x] 1.4 Commit and update memory bank — merged via PR #1 from dev-ayush-17/ml

- 🔄 **M2: Model Training & Serialization** — Nearly complete (1 gap)
  - [x] 2.1 Implement train.py — all 5 models trained (per git log: "Implememnted all ML algorithms")
  - [x] 2.2 Evaluation (metrics.json) — metrics.json exists with all model metrics
  - [x] 2.3 Serialization (5 .pkl model files) — 4 of 5 serialized:
    - [x] logistic_regression.pkl
    - [x] knn.pkl
    - [x] svm.pkl
    - [x] naive_bayes.pkl
    - [ ] decision_tree.pkl — **MISSING, needs resolution**
  - [ ] 2.4 Sanity check (load + predict on sample row) — not verified on disk
  - [ ] 2.5 Commit and update memory bank — pending

- 📋 **M3: Backend API** — Not started (ready to begin)
  - [ ] 3.1 Project setup (FastAPI app, requirements.txt)
  - [ ] 3.2 Model loading at startup
  - [ ] 3.3 POST /api/v1/predict
  - [ ] 3.4 GET /api/v1/metrics
  - [ ] 3.5 GET /api/v1/form-schema
  - [ ] 3.6 GET /api/v1/health
  - [ ] 3.7 Input validation (422/400 errors)
  - [ ] 3.8 CORS configuration
  - [ ] 3.9 Commit and update memory bank

- 📋 **M4: Frontend** — Not started
  - [ ] 4.1 Prediction page (form + 5-model results)
  - [ ] 4.2 Report Card page (metrics table)
  - [ ] 4.3 Charts (bar charts for comparison)
  - [ ] 4.4 Trade-off narrative callouts
  - [ ] 4.5 Navigation between pages
  - [ ] 4.6 Visual polish (LinkedIn-ready)
  - [ ] 4.7 Commit and update memory bank

- 📋 **M5: Deployment** — Not started
  - [ ] 5.1 Deployment config (Dockerfile or platform config)
  - [ ] 5.2 Deploy to public URL
  - [ ] 5.3 Test deployed version
  - [ ] 5.4 Commit and update memory bank

- 📋 **M6: LinkedIn Write-Up Assets** — Not started
  - [ ] 6.1 Screenshots (prediction page + report card)
  - [ ] 6.2 README polish
  - [ ] 6.3 LinkedIn post draft
  - [ ] 6.4 Final memory bank update

## Completed Items

- ✅ PRD.md written (v1.0.0, 1683 lines)
- ✅ Memory bank bootstrapped (all 4 files: activeContext, progress, decisions, projectbrief)
- ✅ Dataset confirmed: Online Shoppers Purchasing Intention (UCI ID 468) — recorded in decisions.md
- ✅ `docs/ml-guide/` created (10 guide files: 00–09)
- ✅ M1 complete: preprocessing.py → scaler.pkl, encoder.pkl, feature_config.json
- ✅ M2 nearly complete: 4/5 models serialized + metrics.json (decision_tree.pkl missing)

## Key Dates

- **2026-09-19:** Project initialized. PRD written. Memory bank bootstrapped. Dataset confirmed. ML guide index created. Git initialized.
- **2026-09-21:** ML pipeline merged via PR #1 from dev-ayush-17/ml. Artifacts on disk: 4/5 models + transformers + metrics.json. Decision Tree .pkl missing. Memory bank updated to reflect real state.
