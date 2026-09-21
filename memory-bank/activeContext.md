# Active Context

## Current Focus
Pre-implementation. PRD is complete (v1.0.0). Memory bank is bootstrapped. Dataset confirmed. No implementation code exists yet.

## What Exists
- `PRD.md` — full specification (1683 lines)
- `memory-bank/` — all 4 files (activeContext, progress, decisions, projectbrief)
- `docs/ml-guide/00-workflow-and-git-map.md` — ML guide index & git workflow map (branch strategy, commit conventions, guide TOC)
- `CLAUDE.md` — agent rules
- Git: 1 initial commit (`702b28d chore: initial commit`), `docs/` untracked

## What Does NOT Exist Yet
- `model_training/` directory — no data, no preprocessing.py, no train.py, no artifacts
- `backend/`, `frontend/` directories
- `requirements.txt`, `.gitignore`
- Any dataset, notebook, or serialized model

## Immediate Next Steps
1. Download the Online Shoppers Purchasing Intention dataset (UCI ID 468) into `model_training/data/raw/`.
2. Create EDA notebook (`model_training/notebooks/01_eda.ipynb`) — inspect shape, dtypes, distributions, missing values, correlations.
3. Implement `model_training/preprocessing.py` — cleaning, encoding, scaling, splitting per PRD Section 7.
4. Resolve OQ2 (encoding strategy) before implementing the encoder — default is OneHotEncoder for all models.
5. Create the remaining ML guide docs (01–09) as implementation progresses, per the index in `00-workflow-and-git-map.md`.

## Blockers
None. Dataset is confirmed. OQ2–OQ10 still open — check `decisions.md` before resolving any.

## Session Notes
- **2026-09-19:** PRD created (v1.0.0). Memory-bank bootstrapped with all 4 files. Dataset confirmed as **Online Shoppers Purchasing Intention** (UCI ID 468). `docs/ml-guide/00-workflow-and-git-map.md` created with branch strategy, commit conventions, and guide TOC. Git initialized with one initial commit. No implementation code written yet.
