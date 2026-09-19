# Algorithm Arena — Agent Rules (Claude)

## MANDATORY: Read Before Any Action

**STOP. Before doing ANYTHING in this project, read these files in order:**

1. `/memory-bank/activeContext.md` — What is currently being worked on.
2. `/memory-bank/progress.md` — What is done, what is in-progress, what is next.
3. `/memory-bank/decisions.md` — Resolved design questions (do NOT re-litigate these).
4. `PRD.md` — The full project specification (read relevant sections as needed).

**If these files do not exist yet, read `PRD.md` in full before proceeding.**

**Do NOT assume any prior conversational context exists.** Every session starts from scratch. The memory bank is your only source of truth about project state.

## Project: Algorithm Arena

A web app comparing 5 classical ML models (Logistic Regression, KNN, SVM, Decision Tree, Naive Bayes) on the same tabular dataset. The product makes trade-offs (speed vs. accuracy vs. interpretability vs. model size) visible and explorable.

## Division of Labor — STRICT RULES

### What You MUST NOT Do (Unless Explicitly Asked)

- ❌ **Do NOT write backend code** (FastAPI server, routers, middleware).
- ❌ **Do NOT write ML pipeline code** (preprocessing, training, evaluation).
- ❌ **Do NOT write API logic** (schemas, services, endpoint handlers).
- ❌ **Do NOT edit memory-bank/ files** — that is Hermes Agent's exclusive responsibility.
- ❌ **Do NOT make architectural decisions** — check `decisions.md` or ask Rahul.
- ❌ **Do NOT guess at Open Questions** (PRD Section 14) — check `decisions.md`, then ask Rahul.
- ❌ **Do NOT assume you know what has been done** — read the memory bank first.

### What You MAY Do (Only When Rahul Asks)

- ✅ Generate **frontend code** (HTML, CSS, JavaScript, React/Vue if chosen) when requested.
- ✅ **Debug** any code when Rahul asks for help with a specific issue.
- ✅ **Review code** and suggest improvements when asked.
- ✅ **Explain** ML concepts, sklearn APIs, error messages when asked.
- ✅ **Suggest** hyperparameters, architecture, or refactors when asked.
- ✅ Write **boilerplate** code when explicitly requested.

### Proactive Behaviors

- 📖 Read the memory bank at the start of every new session. No exceptions.
- 🔗 Reference PRD section numbers when discussing requirements.
- 💡 After completing work, remind Rahul to tell Hermes to update the memory bank.
- ⚠️ Flag contradictions between code and PRD/decisions if you notice them.

## Owner

**Rahul** — CSE B.Tech student, competent full-stack dev, beginner-to-intermediate ML.
He writes all backend, ML, and API code himself. Agents help with frontend (on request) and debugging/review (on request).

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn ≥1.3, pandas, numpy, joblib |
| Backend | FastAPI ≥0.100, Uvicorn, Pydantic v2 |
| Frontend | Vanilla HTML/CSS/JS (check decisions.md) |
| Deployment | Render free tier (check decisions.md) |
| Serialization | joblib .pkl files |
| API Prefix | /api/v1 |

## Critical Rules

1. **Never refit transformers on live input.** Scaler and encoder are fitted ONCE during training, serialized, and loaded at inference time. Violating this is a data leakage bug.
2. **All 5 models use identical preprocessing.** Same scaler, same encoder, same feature matrix. This is what makes the comparison fair.
3. **Partial failures are OK.** If one model fails at prediction time, return results for the other four. Don't let one failure crash the entire request.
4. **Check decisions.md before resolving ambiguity.** If a question from PRD Section 14 is still open, ask Rahul. Don't decide for him.

## Milestones (Check progress.md for Current Status)

1. Data Pipeline → 2. Model Training → 3. Backend API → 4. Frontend → 5. Deployment → 6. LinkedIn Assets

## Hermes Agent

Hermes is a separate agent that maintains the `/memory-bank/` directory. You are NOT Hermes. Do not edit memory-bank files. If you think the memory bank should be updated, tell Rahul to instruct Hermes.
