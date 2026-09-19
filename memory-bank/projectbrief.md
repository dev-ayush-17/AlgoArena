# Algorithm Arena — Project Brief

## What It Is
Algorithm Arena is a live, deployable web application that trains five classical ML algorithms — Logistic Regression, KNN, SVM, Decision Tree, and Naive Bayes — on the same tabular dataset, then lets a user submit a data point through a web form and see, side by side: each model's prediction, its confidence, and its inference latency — plus a separate "Report Card" view comparing accuracy, precision, recall, F1, model size, and inference time across all five.

## Why It Exists
1. Solidify classical ML fundamentals through a full build, not just notebooks.
2. Practice full-stack integration of ML into a real served product.
3. Produce something LinkedIn-presentable — the differentiator is the trade-off narrative (speed vs. accuracy vs. interpretability), not the algorithms themselves.
4. Deliberately practice AI-assisted / agentic coding discipline across multiple tools.

## Owner
**Rahul** — CSE B.Tech student, competent full-stack developer, beginner-to-intermediate ML.
- Writes all backend, ML pipeline, and API code.
- Agents may scaffold frontend code on request.
- Hermes Agent maintains this memory-bank (standing role).

## Tech Stack
- **ML:** Python 3.11+, scikit-learn ≥1.3, pandas, numpy, joblib
- **Backend:** FastAPI ≥0.100, Uvicorn, Pydantic v2
- **Frontend:** Vanilla HTML/CSS/JS (pending — check decisions.md)
- **Deployment:** Render free tier (pending — check decisions.md)

## Key Principle
**The point is the TRADE-OFF NARRATIVE.** No single model wins on every dimension. A model with 86% accuracy might just be predicting the majority class every time. KNN might be the most accurate but 100x slower and 3000x larger. Algorithm Arena makes these trade-offs visible and explorable.

## Dataset
Pending confirmation — see PRD.md Section 4 for candidates.

## Full Specification
See `PRD.md` for the complete Product Requirements Document (~1000+ lines).

## Milestones
1. Data Pipeline
2. Model Training & Serialization
3. Backend API
4. Frontend
5. Deployment
6. LinkedIn Write-Up Assets

See `progress.md` for current status.
