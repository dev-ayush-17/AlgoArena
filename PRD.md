# Algorithm Arena — Product Requirements Document

> **Document version:** 1.0.0
> **Last updated:** 2026-09-19
> **Author:** Rahul (owner) + PRD drafted by AI architect
> **Status:** ACTIVE — this is the canonical source of truth for the project.

---

## Table of Contents

1. [Overview & Problem Statement](#1-overview--problem-statement)
2. [Goals / Non-Goals](#2-goals--non-goals)
3. [Division of Labor](#3-division-of-labor)
4. [Dataset Selection](#4-dataset-selection)
5. [System Architecture](#5-system-architecture)
6. [Repository / Folder Structure](#6-repository--folder-structure)
7. [Data & Feature Pipeline Spec](#7-data--feature-pipeline-spec)
8. [Model Training Spec](#8-model-training-spec)
9. [API Contract](#9-api-contract)
10. [Frontend Requirements](#10-frontend-requirements)
11. [Milestones / Build Order](#11-milestones--build-order)
12. [Edge Cases & Failure Modes](#12-edge-cases--failure-modes)
13. [Deployment Plan](#13-deployment-plan)
14. [Open Questions](#14-open-questions)
15. [Glossary / Context for Future Agents](#15-glossary--context-for-future-agents)
16. [Cross-Session Memory & Agent Handoff Protocol](#16-cross-session-memory--agent-handoff-protocol)

---

## 1. Overview & Problem Statement

**Algorithm Arena** is a live, deployable web application that trains five classical machine-learning algorithms — Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Decision Tree, and Naive Bayes — on the *same* tabular dataset, then exposes a web interface where a user can submit a single data point through a form and see, side by side, each model's prediction, its confidence score, and its per-inference latency in milliseconds. A separate "Report Card" view compares all five models on accuracy, precision, recall, F1-score, serialized model file size, and average inference time — making the *trade-offs* between models visible and explorable, not just surfacing one accuracy number.

The product exists to demonstrate that model selection is an engineering trade-off, not a "pick the highest accuracy" exercise, and to serve as a compelling, LinkedIn-presentable portfolio piece that goes beyond the typical "trained a model in a notebook" project.

---

## 2. Goals / Non-Goals

### Goals

| # | Goal | Why It Matters |
|---|------|---------------|
| G1 | Solidify classical ML fundamentals through a **full build**, not just notebooks. | Rahul's learning objective — connect theory to a served product. |
| G2 | Practice **full-stack integration** of ML into a real served product (data pipeline → serialized models → API → frontend). | Bridges the gap between ML experimentation and software engineering. |
| G3 | Produce something **genuinely LinkedIn-presentable** — the differentiator is the trade-off narrative (speed vs. accuracy vs. interpretability), not the algorithms themselves. | Portfolio impact; demonstrates systems thinking, not just sklearn usage. |
| G4 | Deliberately practice **AI-assisted / agentic coding discipline** across multiple tools and models, including a persistent memory agent (Hermes Agent). | Meta-skill: learning to orchestrate AI agents effectively, with clear ownership boundaries. |
| G5 | Compare five models on identical data with **identical preprocessing** so the comparison is fair and reproducible. | Scientific rigor; the product's credibility depends on apples-to-apples comparison. |
| G6 | Capture and display **inference latency per model** alongside accuracy metrics. | This is the core "trade-off" narrative — a fast, less-accurate model may be preferable in some contexts. |
| G7 | Serialize the entire pipeline (fitted transformers + models) so inference is **deterministic and never refits** on live data. | Production correctness — data leakage at inference time is a common junior mistake. |

### Non-Goals

| # | Non-Goal | Reasoning |
|---|----------|-----------|
| NG1 | Deep learning / neural networks. | Out of scope — this project is about classical ML fundamentals. |
| NG2 | From-scratch algorithm implementation (writing gradient descent, distance calculations, etc.). | sklearn-level implementation is the intended depth per Rahul's skill level and goals. |
| NG3 | Production-grade authentication, authorization, or security hardening. | This is a portfolio project, not a multi-tenant SaaS. Basic input validation is in scope; OAuth/JWT is not. |
| NG4 | Multi-user accounts, user data persistence, or session management. | Single-user demo app. No database for user state. |
| NG5 | AutoML or hyperparameter optimization (GridSearchCV, Optuna, etc.). | Manual/simple hyperparameter choices are fine. The focus is on comparison, not squeezing +0.5% accuracy. |
| NG6 | Real-time model retraining or online learning. | Models are trained offline, serialized, and served statically. |
| NG7 | Mobile-native apps. | Web-only. Responsive design is nice-to-have but not required. |
| NG8 | CI/CD pipelines or automated testing infrastructure. | Nice-to-have in a later phase, not part of the MVP. |

---

## 3. Division of Labor

This project has **one human owner (Rahul)** and **multiple AI agents** with carefully scoped roles. The table below is the authoritative reference for who does what.

| Component | Owner | What the owner does | What an agent may do **unprompted** | What an agent may do **only when explicitly asked** |
|-----------|-------|--------------------|------------------------------------|-----------------------------------------------------|
| **Backend (FastAPI server)** | Rahul | Writes all server code: main app, routers, middleware, error handling. | Nothing. | Debug assistance, code review, suggest improvements. |
| **ML Pipeline (preprocessing, training, evaluation, serialization)** | Rahul | Writes all pipeline code: `preprocessing.py`, `train.py`, evaluation scripts, notebook exploration. | Nothing. | Debug assistance, code review, explain sklearn API behavior, suggest hyperparameters. |
| **API Logic (endpoint handlers, request/response schemas)** | Rahul | Writes all Pydantic schemas, service layers, prediction logic. | Nothing. | Debug assistance, code review, validate schema design. |
| **Frontend (UI pages, components, styling)** | Agent (on request) | — | Nothing. Agents do NOT spontaneously generate frontend code. | Scaffold pages, generate components, write CSS/styling, build the full frontend when Rahul asks. |
| **Memory-Bank Maintenance** | Hermes Agent (standing role) | — | Nothing automatic. Hermes updates memory-bank files **only when explicitly told** by Rahul ("update the memory bank") after a confirmed unit of work. | Read filesystem/git state, then write/update `activeContext.md` and `progress.md`. See Section 16 for full protocol. |
| **Code Review / Debugging Assistance** | Any agent (on request only) | — | Nothing. | Review code for bugs, suggest fixes, explain error messages, propose refactors. |
| **Documentation (README, inline docs)** | Rahul | Writes project README and inline documentation. | Nothing. | Suggest README sections, proofread documentation. |
| **Deployment Configuration** | Rahul | Writes Dockerfiles, deployment configs, environment setup. | Nothing. | Suggest deployment strategies, debug deployment issues. |

> **Key principle:** No agent writes backend, ML, or API code unless Rahul explicitly asks for help with a specific piece. Frontend is the one exception where agents may generate full code — but still only on request, never proactively.

---

## 4. Dataset Selection

### Selection Criteria

The dataset must satisfy ALL of the following:

- **Publicly available** with a stable download link (UCI ML Repository, OpenML, or Kaggle).
- **Between 1,000 and 50,000 rows** — large enough for meaningful model comparison, small enough that training is fast on a laptop CPU.
- **8 to 30 features** — enough complexity for a non-trivial pipeline, not so many that feature engineering becomes the whole project.
- **Binary or small multi-class target** (2–5 classes) — compatible with all five algorithms without special handling.
- **Mix of numerical and categorical features** — ensures the preprocessing pipeline is non-trivial (encoding + scaling both required).
- **Interesting domain** — makes the LinkedIn narrative compelling.
- **NOT Titanic, Iris, MNIST, Wine, or any generic "customer churn" dataset** that appears in every beginner tutorial.

### Candidate Datasets

#### Candidate 1: **CDC Diabetes Health Indicators** ⭐ RECOMMENDED DEFAULT

| Attribute | Detail |
|-----------|--------|
| **Source** | [UCI ML Repository — CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) / Also available on [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) |
| **Size** | ~253,680 rows × 22 features (can be subsampled to ~20,000–30,000 for tractable training) |
| **Target Variable** | `Diabetes_binary` — binary classification (0 = no diabetes, 1 = diabetes/prediabetes) |
| **Feature Mix** | Mix of binary indicators (e.g., HighBP, HighChol, Smoker), ordinal (e.g., GenHlth 1-5, Age category 1-13), and continuous (BMI, MentHlth days, PhysHlth days). |
| **Class Distribution** | Imbalanced (~86% negative, ~14% positive in full set) — provides an opportunity to discuss class imbalance as a trade-off dimension. |
| **Why It's Differentiated** | Healthcare domain is compelling for storytelling. Class imbalance makes the precision/recall trade-off genuinely interesting (a model with 86% accuracy could just be predicting "no diabetes" every time). This forces the Report Card to reveal something non-obvious. BMI, blood pressure, lifestyle factors are relatable to any audience. |
| **License** | CC BY 4.0 — free to use with attribution. |

> **⚠️ PENDING CONFIRMATION:** This dataset is recommended as the default. Rahul must confirm before any data pipeline work begins. If rejected, fall back to Candidate 2 or 3.

#### Candidate 2: **Heart Disease (Cleveland — UCI)**

| Attribute | Detail |
|-----------|--------|
| **Source** | [UCI ML Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) / [Kaggle mirror](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) |
| **Size** | ~920 rows × 16 features (combined Cleveland + other sites can reach ~1,025 rows) |
| **Target Variable** | `num` — originally 0-4 severity, commonly binarized to 0 = no disease, 1 = disease. |
| **Feature Mix** | Numerical (age, trestbps, chol, thalach, oldpeak) + Categorical (sex, cp, fbs, restecg, exang, slope, thal). |
| **Class Distribution** | Roughly balanced (~46% positive, ~54% negative in Cleveland set). |
| **Why It's Differentiated** | Classic but not *overused* in the same way Titanic/Iris are. Medical domain. Small enough for instant training. However, the small size limits the statistical robustness of metric comparisons. |
| **License** | CC BY 4.0. |

#### Candidate 3: **Adult Income (Census Income)**

| Attribute | Detail |
|-----------|--------|
| **Source** | [UCI ML Repository — Adult/Census Income](https://archive.ics.uci.edu/dataset/2/adult) / [OpenML](https://www.openml.org/d/1590) |
| **Size** | ~48,842 rows × 14 features. |
| **Target Variable** | `income` — binary (<=50K vs >50K). |
| **Feature Mix** | Strong mix: numerical (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week) + categorical (workclass, education, marital-status, occupation, relationship, race, sex, native-country). |
| **Class Distribution** | Imbalanced (~76% <=50K, ~24% >50K). |
| **Why It's Differentiated** | Good feature diversity (high-cardinality categoricals like `native-country` and `occupation` add pipeline complexity). Income prediction is universally relatable. However, it *is* relatively well-known in ML circles, though less "tutorial cliché" than Titanic/Iris. |
| **License** | CC BY 4.0. |

### Additional Candidates (from Research)

#### Candidate 4: **Online Shoppers Purchasing Intention** ⭐ STRONG ALTERNATIVE

| Attribute | Detail |
|-----------|--------|
| **Source** | [UCI ML Repository — Dataset ID 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (DOI: `10.24432/C5F88Q`) |
| **Size** | 12,330 rows × 18 columns (17 features + 1 target). |
| **Target Variable** | `Revenue` — binary (True = purchase completed, False = no purchase). |
| **Feature Mix** | **10 numerical** (page durations, bounce/exit rates, page values, special day) + **7 categorical/boolean** (Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend). |
| **Class Distribution** | Imbalanced — 84.5% no purchase, 15.5% purchase (~5.5:1 ratio). Natural real-world conversion rate. |
| **Why It's Differentiated** | **E-commerce CRO (Conversion Rate Optimization) is a universally compelling domain.** Key ML-pedagogical properties: (1) Feature scaling disparities — page durations range 0–60,000+ while rates are bounded [0,1], so KNN/SVM fail without scaling while Decision Trees are immune → directly illustrates trade-offs. (2) `BounceRates` and `ExitRates` have correlation >0.90, violating Naive Bayes's independence assumption → exposes GaussianNB weakness. (3) Zero-inflated features (`PageValues`) create non-linearities where trees/RBF-SVM outshine linear models. (4) Goldilocks size: SVM trains in 2-5 seconds vs. 45+ seconds on larger datasets. |
| **License** | CC BY 4.0 (Sakar et al., 2018). |
| **Quick Load** | `from ucimlrepo import fetch_ucirepo; dataset = fetch_ucirepo(id=468)` |

#### Candidate 5: **Default of Credit Card Clients**

| Attribute | Detail |
|-----------|--------|
| **Source** | [UCI ML Repository — Dataset ID 350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (DOI: `10.24432/C55S3H`) |
| **Size** | 30,000 rows × 24 columns (23 features + 1 target). |
| **Target Variable** | `default.payment.next.month` — binary (1 = default, 0 = non-default). |
| **Feature Mix** | **Continuous financial** (credit limit, age, 6 months of bill statements, 6 months of repayment amounts) + **Categorical/ordinal** (sex, education, marriage, 6 months of payment delay history on ordinal scale -2 to +9). |
| **Class Distribution** | Moderately imbalanced — 77.9% non-default, 22.1% default (~3.5:1 ratio). |
| **Why It's Differentiated** | FinTech/credit risk domain signals enterprise maturity. At 30K rows, it stress-tests SVM and KNN ($O(N^2)$ complexity) while Decision Trees and Naive Bayes remain fast — excellent for latency trade-off narrative. Scale ranges from $10,000 to $1,000,000+ contrast sharply with ordinal codes, demanding robust pipeline design. |
| **License** | CC BY 4.0 (Yeh & Lien, 2009). |
| **Quick Load** | `from ucimlrepo import fetch_ucirepo; dataset = fetch_ucirepo(id=350)` |

### Recommendation Summary

| Criterion | Online Shoppers ⭐ | CDC Diabetes | Heart Disease | Adult Income | Credit Card Default |
|-----------|-------------------|--------------|---------------|-------------|---------------------|
| Size (sweet spot) | ✅ 12,330 — ideal | ⚠️ 254K, needs subsampling | ❌ Very small (~920) | ⚠️ ~49K, SVM slow | ⚠️ 30K, SVM moderate |
| Feature diversity | ✅ Strong mix | ✅ Mixed types | ✅ Mixed types | ✅ High cardinality | ✅ Financial + ordinal |
| Class imbalance | ✅ 84.5/15.5 — great for P/R | ✅ 86/14 — great | ❌ Balanced | ✅ 76/24 — moderate | ✅ 78/22 — moderate |
| Domain appeal (LinkedIn) | ✅ E-commerce CRO — universal | ✅ Healthcare | ✅ Healthcare | ⚠️ Economics | ✅ FinTech |
| Novelty (not overused) | ✅ Uncommon in tutorials | ✅ Uncommon | ⚠️ Somewhat common | ⚠️ Moderately common | ⚠️ Somewhat common |
| Exposes model trade-offs | ✅ Scaling, correlation, non-linearity | ✅ Class imbalance | ⚠️ Too small for robust comparison | ✅ High-cardinality encoding | ✅ Latency stress test |
| Training speed | ✅ 2-5s per model | ⚠️ Needs subsampling | ✅ Instant | ⚠️ SVM slow at 49K | ⚠️ SVM ~15-45s |

**Default recommendation: Online Shoppers Purchasing Intention (UCI ID 468).** Three key reasons:
1. **Goldilocks size** (12,330 rows) — all 5 models train in seconds, no subsampling needed.
2. **Pedagogically exposes model strengths/weaknesses** — the correlated features directly violate Naive Bayes assumptions, the scaling disparities punish un-scaled KNN/SVM, and the zero-inflated features reward non-linear models. These are exactly the trade-offs Algorithm Arena is designed to surface.
3. **LinkedIn narrative** — "Benchmarking 5 ML models to predict e-commerce purchase intent" is immediately compelling to any recruiter or hiring manager.

**Runner-up: CDC Diabetes Health Indicators** — equally strong for the class-imbalance narrative, but requires subsampling from 254K rows.

> **✅ CONFIRMED (2026-09-19):** Online Shoppers Purchasing Intention (UCI ID 468) selected as the project dataset. Decision recorded in `/memory-bank/decisions.md`.

---

## 5. System Architecture

The system has two distinct runtime contexts — **Offline Training** and **Online Serving** — plus a non-runtime **Memory Bank** for project metadata.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           OFFLINE TRAINING PIPELINE                             │
│                        (runs locally, on demand, by Rahul)                       │
│                                                                                 │
│  ┌──────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌───────────┐  │
│  │ Raw CSV  │───▶│ preprocessing.py │───▶│    train.py      │───▶│ Artifacts │  │
│  │ (data/)  │    │                  │    │                  │    │ (artifacts│  │
│  │          │    │ • Load CSV       │    │ • Load processed │    │  folder)  │  │
│  │          │    │ • Clean/impute   │    │   data           │    │           │  │
│  │          │    │ • Encode cats    │    │ • Train 5 models │    │ • *.pkl   │  │
│  │          │    │ • Scale nums     │    │ • Evaluate each  │    │   models  │  │
│  │          │    │ • Split train/   │    │ • Measure latency│    │ • scaler  │  │
│  │          │    │   test           │    │ • Save models +  │    │   .pkl    │  │
│  │          │    │ • Fit & save     │    │   metrics        │    │ • encoder │  │
│  │          │    │   transformers   │    │                  │    │   .pkl    │  │
│  │          │    │                  │    │                  │    │ • metrics │  │
│  │          │    │                  │    │                  │    │   .json   │  │
│  └──────────┘    └──────────────────┘    └──────────────────┘    └───────────┘  │
│                                                                                 │
│  Output: /model_training/artifacts/ directory populated with serialized         │
│          models, fitted transformers, and evaluation metrics.                    │
└─────────────────────────────────────────────────────────────────────────────────┘

                              │
                              │  Artifacts are loaded at server startup
                              ▼

┌─────────────────────────────────────────────────────────────────────────────────┐
│                            ONLINE SERVING APP                                   │
│                     (FastAPI backend + Static frontend)                          │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                          FastAPI Backend                                   │  │
│  │                                                                            │  │
│  │  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────────┐   │  │
│  │  │   Startup   │    │  POST /predict   │    │   GET /metrics          │   │  │
│  │  │   Event     │    │                  │    │                          │   │  │
│  │  │             │    │ 1. Validate input│    │ 1. Load metrics.json    │   │  │
│  │  │ Load all    │    │ 2. Transform via │    │ 2. Return all 5 models' │   │  │
│  │  │ .pkl files  │    │    saved scaler/ │    │    accuracy, precision, │   │  │
│  │  │ into memory │    │    encoder       │    │    recall, F1, latency, │   │  │
│  │  │             │    │ 3. Run all 5     │    │    model file size      │   │  │
│  │  │ Load        │    │    models        │    │                          │   │  │
│  │  │ metrics.json│    │ 4. Time each     │    └──────────────────────────┘   │  │
│  │  │             │    │    inference     │                                    │  │
│  │  └─────────────┘    │ 5. Return 5      │    ┌──────────────────────────┐   │  │
│  │                      │    predictions + │    │   GET /form-schema      │   │  │
│  │                      │    confidences + │    │                          │   │  │
│  │                      │    latencies     │    │ Returns feature names,  │   │  │
│  │                      └─────────────────┘    │ types, valid ranges/    │   │  │
│  │                                              │ categories for the      │   │  │
│  │                                              │ frontend form builder   │   │  │
│  │                                              └──────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                           Frontend (SPA)                                   │  │
│  │                                                                            │  │
│  │  ┌───────────────────────┐    ┌─────────────────────────────────────────┐  │  │
│  │  │   Prediction Page     │    │          Report Card Page               │  │  │
│  │  │                       │    │                                         │  │  │
│  │  │ • Input form (built   │    │ • Table: 5 models × 6 metrics          │  │  │
│  │  │   from /form-schema)  │    │ • Bar charts for visual comparison     │  │  │
│  │  │ • Submit → POST       │    │ • Highlight best/worst per metric      │  │  │
│  │  │   /predict            │    │ • Trade-off narrative callouts          │  │  │
│  │  │ • Results: 5 cards,   │    │                                         │  │  │
│  │  │   each showing:       │    │                                         │  │  │
│  │  │   - Model name        │    │                                         │  │  │
│  │  │   - Prediction        │    │                                         │  │  │
│  │  │   - Confidence %      │    │                                         │  │  │
│  │  │   - Latency (ms)      │    │                                         │  │  │
│  │  └───────────────────────┘    └─────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                        MEMORY BANK (Project Metadata)                           │
│                     NOT part of any runtime system.                              │
│             Lives at /memory-bank/ in the repo root.                            │
│                                                                                 │
│  ┌─────────────────────┐  ┌────────────────────┐  ┌─────────────────────────┐  │
│  │  projectbrief.md    │  │ activeContext.md   │  │  progress.md            │  │
│  │                     │  │                    │  │                         │  │
│  │  Mirrors PRD core   │  │  What's being      │  │  Running log:           │  │
│  │  facts. Rarely      │  │  worked on right   │  │  ✅ Done                │  │
│  │  changes.           │  │  now. Hermes       │  │  🔄 In-progress        │  │
│  │                     │  │  updates this.     │  │  📋 Next                │  │
│  └─────────────────────┘  └────────────────────┘  └─────────────────────────┘  │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  decisions.md                                                            │   │
│  │                                                                          │   │
│  │  Key architectural/design choices + rationale, so nothing gets           │   │
│  │  re-litigated in future sessions.                                        │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘

          ▲                          ▲
          │                          │
    Hermes reads                 Any coding agent
    project state &              reads these files
    updates memory-bank          FIRST before doing
    when told to.                any work.
```

### Key Architectural Decisions

1. **Offline/Online split:** Training never happens at request time. All models are pre-trained, serialized, and loaded into memory once at server startup.
2. **Shared transformers:** A single fitted `StandardScaler` and a single fitted encoder (e.g., `OrdinalEncoder` or `OneHotEncoder`) are shared across all five models. This guarantees identical input representation — a fair comparison.
3. **In-memory model serving:** All five `.pkl` files are loaded into a Python dict at startup. No disk I/O on the prediction hot path.
4. **Stateless API:** The backend has no database. Every request is independent. `GET /metrics` reads from a pre-computed JSON file.
5. **Memory Bank is metadata:** The `/memory-bank/` directory is *about* the project (what's done, what's next, key decisions). It is not read by the application at runtime.

---

## 6. Repository / Folder Structure

```
AlgoArena/                          # Repository root
│
├── PRD.md                          # THIS FILE — canonical source of truth
├── README.md                       # Public-facing project description
├── .gitignore                      # See Section 13 for what's ignored
├── .antigravityrules               # Rules for Antigravity/Gemini agents (see Sec 16)
├── CLAUDE.md                       # Rules for Claude-based agents (see Sec 16)
├── requirements.txt                # Python dependencies (pinned versions)
│
├── memory-bank/                    # Project metadata for cross-session continuity
│   ├── projectbrief.md             # Core facts from PRD, rarely changes
│   ├── activeContext.md            # Current work focus — Hermes updates this
│   ├── progress.md                 # Done / In-progress / Next — Hermes updates this
│   └── decisions.md                # Architectural decisions + rationale
│
├── model_training/                 # OFFLINE: All ML pipeline code and artifacts
│   ├── data/                       # Raw and processed data files
│   │   ├── raw/                    # Original downloaded CSV(s) — gitignored if large
│   │   └── processed/              # Cleaned/split data (optional intermediate storage)
│   │
│   ├── notebooks/                  # Jupyter notebooks for EDA and experimentation
│   │   └── 01_eda.ipynb            # Exploratory data analysis
│   │
│   ├── preprocessing.py            # Data loading, cleaning, encoding, scaling, splitting
│   ├── train.py                    # Train all 5 models, evaluate, serialize
│   ├── evaluate.py                 # (Optional) Standalone evaluation/comparison script
│   │
│   └── artifacts/                  # OUTPUT of training — loaded by backend at startup
│       ├── logistic_regression.pkl # Serialized Logistic Regression model
│       ├── knn.pkl                 # Serialized KNN model
│       ├── svm.pkl                 # Serialized SVM model
│       ├── decision_tree.pkl       # Serialized Decision Tree model
│       ├── naive_bayes.pkl         # Serialized Naive Bayes model
│       ├── scaler.pkl              # Fitted StandardScaler
│       ├── encoder.pkl             # Fitted categorical encoder
│       ├── feature_config.json     # Feature names, types, valid ranges/categories
│       └── metrics.json            # All 5 models' evaluation metrics
│
├── backend/                        # ONLINE: FastAPI application
│   ├── main.py                     # FastAPI app instantiation, startup events, CORS
│   ├── config.py                   # Configuration (paths, environment variables)
│   │
│   ├── routers/                    # API route definitions
│   │   ├── __init__.py
│   │   ├── predict.py              # POST /predict endpoint
│   │   ├── metrics.py              # GET /metrics endpoint
│   │   └── schema.py               # GET /form-schema endpoint
│   │
│   ├── schemas/                    # Pydantic request/response models
│   │   ├── __init__.py
│   │   ├── predict.py              # PredictionRequest, PredictionResponse, etc.
│   │   └── metrics.py              # MetricsResponse, ModelMetrics, etc.
│   │
│   ├── services/                   # Business logic layer
│   │   ├── __init__.py
│   │   ├── model_service.py        # Model loading, prediction orchestration
│   │   └── preprocessing_service.py # Input transformation using saved transformers
│   │
│   └── utils/                      # Shared utilities
│       ├── __init__.py
│       └── timing.py               # Inference timing utilities
│
└── frontend/                       # Static frontend (SPA)
    ├── index.html                  # Entry point
    ├── css/
    │   └── styles.css              # Application styles
    ├── js/
    │   ├── app.js                  # Main application logic
    │   ├── predict.js              # Prediction page logic
    │   └── report.js               # Report card page logic
    └── assets/                     # Static assets (icons, images if any)
```

### Notes on Folder Structure

- **`model_training/artifacts/`** is the bridge between offline and online. The backend reads from this directory; the training pipeline writes to it.
- **`memory-bank/`** is at the repo root, not inside any component directory, because it's about the *project*, not about any single component.
- **Frontend technology:** Vanilla HTML/CSS/JS is the default. If Rahul decides to use React/Vue/Svelte later, the `frontend/` directory structure will change, but the API contract (Section 9) remains stable.
- **No `__pycache__` or `.pyc` files** are committed (handled by `.gitignore`).

---

## 7. Data & Feature Pipeline Spec

This section specifies the exact transformations applied to the raw dataset to produce training-ready features. The golden rule:

> **Every transformer (scaler, encoder) is fitted ONCE on the training set during offline training, serialized to disk, and loaded at inference time. The backend NEVER fits a transformer on live input. Violating this rule introduces data leakage and makes predictions non-deterministic.**

### 7.1 Pipeline Steps (Ordered)

```
Step 1: Load Raw Data
        ↓
Step 2: Initial Cleaning
        ↓
Step 3: Feature Selection / Engineering
        ↓
Step 4: Train/Test Split
        ↓
Step 5: Fit Transformers on Training Set ONLY
        ↓
Step 6: Transform Both Train and Test Sets
        ↓
Step 7: Serialize Fitted Transformers to Disk
        ↓
Step 8: Output Processed Data for Training
```

### 7.2 Step Details

#### Step 1: Load Raw Data

- Load CSV from `model_training/data/raw/`.
- Print shape, dtypes, and first 5 rows for sanity check.
- If the dataset is large (>30,000 rows), take a stratified subsample to ~20,000–25,000 rows to keep training fast. Document the random seed used for reproducibility.

#### Step 2: Initial Cleaning

| Task | Approach |
|------|----------|
| **Missing values — numerical columns** | Impute with **median** (robust to outliers). Use `sklearn.impute.SimpleImputer(strategy='median')`. |
| **Missing values — categorical columns** | Impute with **most frequent** value. Use `SimpleImputer(strategy='most_frequent')`. |
| **Duplicate rows** | Drop exact duplicates. Log how many were removed. |
| **Obvious outliers** | For numerical columns, clip values beyond the 1st and 99th percentiles to those percentile values. Do NOT drop rows — clipping preserves sample size. |
| **Invalid/corrupt values** | Check for impossible values (e.g., negative age, BMI of 0). Replace with NaN, then impute as above. |

#### Step 3: Feature Selection / Engineering

- **Drop columns** that are: identifiers (IDs, names), constant (zero variance), or have >50% missing values.
- **No complex feature engineering** — keep it simple for the MVP. If time permits, consider interaction features or binning, but document any additions in `decisions.md`.
- **Categorize features** into two lists:
  - `numerical_features: List[str]` — columns to be scaled.
  - `categorical_features: List[str]` — columns to be encoded.
- Save these lists to `feature_config.json` along with valid value ranges and categories (used by the frontend form and input validation).

#### Step 4: Train/Test Split

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Test size** | 20% | Standard for datasets of this size. |
| **Stratification** | Stratify on target variable | Preserves class distribution in both splits — critical for imbalanced datasets. |
| **Random state** | 42 | Reproducibility. Hardcoded. |
| **Shuffle** | True (default) | Standard practice. |

```python
# Illustrative — not runnable code
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

#### Step 5: Fit Transformers on Training Set ONLY

| Transformer | Applied To | Implementation | Notes |
|-------------|-----------|----------------|-------|
| **StandardScaler** | All numerical features | `sklearn.preprocessing.StandardScaler` | Fit on `X_train[numerical_features]` only. |
| **OrdinalEncoder** or **OneHotEncoder** | All categorical features | `sklearn.preprocessing.OrdinalEncoder` (default) or `OneHotEncoder(sparse_output=False, handle_unknown='use_encoded_value', unknown_value=-1)` | OrdinalEncoder recommended for tree-based models; OneHotEncoder needed for Logistic Regression/SVM/KNN/NB. See **Decision Point** below. |

**Decision Point — Encoding Strategy:**

Since the five models have different encoding preferences, there are two valid approaches:

- **Option A (Recommended — Simpler):** Use `OneHotEncoder` for all categorical features across all models. Trees can still learn from one-hot encoded features. This gives a single `encoder.pkl` and a uniform feature matrix for all models, keeping the comparison fair.
- **Option B (More nuanced):** Use `ColumnTransformer` with `OrdinalEncoder` for tree-based models and `OneHotEncoder` for linear/distance-based models. This adds complexity and produces model-specific input matrices, making the "same data" claim slightly less clean.

> **⚠️ Recommendation:** Go with Option A (`OneHotEncoder` for all) for simplicity and fairness. Flag in `decisions.md` with rationale. Rahul to confirm.

#### Step 6: Transform Both Train and Test Sets

```python
# Illustrative
X_train_scaled = scaler.transform(X_train[numerical_features])
X_test_scaled = scaler.transform(X_test[numerical_features])
X_train_encoded = encoder.transform(X_train[categorical_features])
X_test_encoded = encoder.transform(X_test[categorical_features])

# Concatenate back
import numpy as np
X_train_processed = np.hstack([X_train_scaled, X_train_encoded])
X_test_processed = np.hstack([X_test_scaled, X_test_encoded])
```

#### Step 7: Serialize Fitted Transformers

```python
# Illustrative
import joblib
joblib.dump(scaler, 'model_training/artifacts/scaler.pkl')
joblib.dump(encoder, 'model_training/artifacts/encoder.pkl')
```

#### Step 8: Output Processed Data

The processed `X_train`, `X_test`, `y_train`, `y_test` arrays are either:
- Returned in-memory to `train.py` (if `preprocessing.py` is imported as a module), OR
- Saved to `model_training/data/processed/` as `.npy` or `.csv` files (if the pipeline is run as separate scripts).

**Recommended approach:** Import `preprocessing.py` functions in `train.py` so data flows in-memory without intermediate file I/O.

### 7.3 Feature Configuration File

The file `model_training/artifacts/feature_config.json` is produced by the preprocessing step and consumed by both the backend (for input validation) and the frontend (for form generation).

```json
{
  "numerical_features": [
    {
      "name": "BMI",
      "dtype": "float64",
      "min": 12.0,
      "max": 98.0,
      "mean": 28.4,
      "description": "Body Mass Index"
    }
  ],
  "categorical_features": [
    {
      "name": "GenHlth",
      "dtype": "int64",
      "categories": [1, 2, 3, 4, 5],
      "description": "General Health (1=Excellent, 5=Poor)"
    }
  ],
  "target": {
    "name": "Diabetes_binary",
    "classes": [0, 1],
    "class_labels": ["No Diabetes", "Diabetes/Prediabetes"]
  },
  "feature_order": ["BMI", "GenHlth", "..."],
  "total_features_after_encoding": 42
}
```

---

## 8. Model Training Spec

All five models are trained on the **identical processed dataset** from Section 7. Each model is evaluated on the **same held-out test set**. This guarantees fair comparison.

### 8.1 Per-Model Specification

#### Model 1: Logistic Regression

| Attribute | Value |
|-----------|-------|
| **Library/Class** | `sklearn.linear_model.LogisticRegression` |
| **Key Hyperparameters** | `max_iter=1000`, `solver='lbfgs'`, `C=1.0` (default regularization), `random_state=42` |
| **Confidence** | `model.predict_proba(X)[0]` — native probability output. |
| **Notes** | Works well with one-hot encoded features. Fast training. Interpretable via coefficients. `max_iter` raised from default 100 to prevent convergence warnings on larger datasets. |

#### Model 2: K-Nearest Neighbors (KNN)

| Attribute | Value |
|-----------|-------|
| **Library/Class** | `sklearn.neighbors.KNeighborsClassifier` |
| **Key Hyperparameters** | `n_neighbors=5` (default, consider 7 or 11 for smoother boundaries), `weights='uniform'`, `metric='minkowski'` (p=2, i.e., Euclidean), `n_jobs=-1` |
| **Confidence** | `model.predict_proba(X)[0]` — proportion of neighbor votes. |
| **Notes** | Inference time scales with dataset size (lazy learner). Feature scaling is critical — already handled by the shared StandardScaler. The serialized file will be large because KNN stores the entire training set. This is a feature, not a bug — it's part of the trade-off narrative (large model size, slow inference, but non-parametric). |

#### Model 3: Support Vector Machine (SVM)

| Attribute | Value |
|-----------|-------|
| **Library/Class** | `sklearn.svm.SVC` |
| **Key Hyperparameters** | `kernel='rbf'`, `C=1.0`, `gamma='scale'`, `probability=True`, `random_state=42` |
| **Confidence** | `model.predict_proba(X)[0]` — enabled via Platt scaling (`probability=True`). |
| **Notes** | `probability=True` is **required** to get confidence scores but adds training overhead (internal cross-validation for Platt scaling). On datasets >20K rows, SVM training can be slow — this is acceptable and is part of the narrative (high accuracy, but expensive). If training exceeds 10 minutes, consider subsampling the training set for SVM only and documenting the decision. |

#### Model 4: Decision Tree

| Attribute | Value |
|-----------|-------|
| **Library/Class** | `sklearn.tree.DecisionTreeClassifier` |
| **Key Hyperparameters** | `max_depth=10` (prevent overfitting), `min_samples_split=5`, `min_samples_leaf=3`, `random_state=42` |
| **Confidence** | `model.predict_proba(X)[0]` — leaf class distribution. |
| **Notes** | Fast training and inference. Highly interpretable (can export tree as text/image). Prone to overfitting without depth constraints. The depth cap is intentionally moderate — deep enough to learn patterns, shallow enough to generalize. Consider exporting a tree visualization as a bonus artifact. |

#### Model 5: Naive Bayes

| Attribute | Value |
|-----------|-------|
| **Library/Class** | `sklearn.naive_bayes.GaussianNB` (if all features are continuous post-encoding) or consider `CategoricalNB` / `BernoulliNB` depending on feature characteristics after encoding. |
| **Key Hyperparameters** | Default (GaussianNB has essentially no hyperparameters). `var_smoothing=1e-9` (default). |
| **Confidence** | `model.predict_proba(X)[0]` — posterior probabilities. |
| **Notes** | Extremely fast training and inference. Strong independence assumption rarely holds in practice. Often the "fastest but least accurate" model — a critical data point in the trade-off narrative. If using `OneHotEncoder` (Option A from Section 7), `GaussianNB` is acceptable; if features are truly binary/categorical, `BernoulliNB` or `CategoricalNB` may be more appropriate. Document choice in `decisions.md`. |

### 8.2 Metrics Captured Per Model

Every model produces the following metrics, computed on the **test set**:

| Metric | How Computed | Format | Purpose |
|--------|-------------|--------|---------|
| **Accuracy** | `sklearn.metrics.accuracy_score(y_test, y_pred)` | float, 0.0–1.0 | Overall correctness. |
| **Precision** | `sklearn.metrics.precision_score(y_test, y_pred, average='weighted')` | float, 0.0–1.0 | Of predicted positives, how many were correct. |
| **Recall** | `sklearn.metrics.recall_score(y_test, y_pred, average='weighted')` | float, 0.0–1.0 | Of actual positives, how many were found. |
| **F1 Score** | `sklearn.metrics.f1_score(y_test, y_pred, average='weighted')` | float, 0.0–1.0 | Harmonic mean of precision and recall. |
| **Avg. Inference Time** | Time 1000 individual `model.predict()` calls, take mean. | float, milliseconds | Per-row prediction speed. |
| **Model File Size** | `os.path.getsize('model.pkl')` after serialization. | float, kilobytes | Storage/deployment cost. |
| **Training Time** | Wall-clock time for `model.fit()`. | float, seconds | Training cost (informational, not shown to end user). |

> **Averaging strategy for precision/recall/F1:** Use `average='weighted'` for multi-class-safe metrics. For binary classification, `average='binary'` also works — document the choice in `decisions.md` and keep it consistent across all five models.

### 8.3 Metrics Output File

All metrics are written to `model_training/artifacts/metrics.json`:

```json
{
  "generated_at": "2026-09-19T00:00:00Z",
  "dataset": "CDC Diabetes Health Indicators",
  "dataset_size": {
    "total_rows": 25000,
    "train_rows": 20000,
    "test_rows": 5000,
    "features": 22,
    "features_after_encoding": 42
  },
  "models": {
    "logistic_regression": {
      "display_name": "Logistic Regression",
      "accuracy": 0.862,
      "precision": 0.841,
      "recall": 0.862,
      "f1_score": 0.847,
      "avg_inference_time_ms": 0.03,
      "model_file_size_kb": 4.2,
      "training_time_seconds": 1.7
    },
    "knn": {
      "display_name": "K-Nearest Neighbors",
      "accuracy": 0.831,
      "precision": 0.812,
      "recall": 0.831,
      "f1_score": 0.818,
      "avg_inference_time_ms": 12.5,
      "model_file_size_kb": 15234.0,
      "training_time_seconds": 0.1
    },
    "svm": {
      "display_name": "Support Vector Machine",
      "accuracy": 0.870,
      "precision": 0.853,
      "recall": 0.870,
      "f1_score": 0.858,
      "avg_inference_time_ms": 1.2,
      "model_file_size_kb": 8750.0,
      "training_time_seconds": 45.3
    },
    "decision_tree": {
      "display_name": "Decision Tree",
      "accuracy": 0.825,
      "precision": 0.810,
      "recall": 0.825,
      "f1_score": 0.815,
      "avg_inference_time_ms": 0.02,
      "model_file_size_kb": 12.8,
      "training_time_seconds": 0.3
    },
    "naive_bayes": {
      "display_name": "Naive Bayes",
      "accuracy": 0.790,
      "precision": 0.775,
      "recall": 0.790,
      "f1_score": 0.778,
      "avg_inference_time_ms": 0.01,
      "model_file_size_kb": 2.1,
      "training_time_seconds": 0.05
    }
  }
}
```

> **⚠️ Note:** The numbers above are ILLUSTRATIVE PLACEHOLDERS. Actual metrics will differ once training is run on the real dataset. The structure is what matters — the values will be populated by `train.py`.

### 8.4 Inference Latency Measurement Protocol

To get meaningful latency numbers:

1. **Warm-up:** Run 100 predictions before measuring (JIT/cache effects).
2. **Measurement:** Time 1,000 individual `model.predict(single_row)` calls using `time.perf_counter()`.
3. **Report:** Mean and standard deviation of those 1,000 measurements, in milliseconds.
4. **At serving time:** Each `/predict` request also times the live inference for each model and returns it in the response. This is the "real" latency the user sees, distinct from the benchmark number in `metrics.json`.

```python
# Illustrative
import time
import numpy as np

times = []
for _ in range(1000):
    start = time.perf_counter()
    model.predict(X_single_row)
    end = time.perf_counter()
    times.append((end - start) * 1000)  # Convert to ms

avg_inference_time_ms = np.mean(times)
std_inference_time_ms = np.std(times)
```

---

## 9. API Contract

The backend exposes a RESTful JSON API via FastAPI. All endpoints are prefixed with `/api/v1`. CORS is enabled for local development (`http://localhost:*`).

### 9.1 Endpoint Summary

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/v1/predict` | Submit a data point, get predictions from all 5 models. |
| `GET` | `/api/v1/metrics` | Get evaluation metrics for all 5 models. |
| `GET` | `/api/v1/form-schema` | Get feature metadata for building the input form. |
| `GET` | `/api/v1/health` | Health check — confirms server is up and models are loaded. |

### 9.2 Detailed Endpoint Specs

#### `POST /api/v1/predict`

Submit a single data point for prediction by all five models.

**Request Body:**

```json
{
  "features": {
    "BMI": 28.5,
    "GenHlth": 3,
    "HighBP": 1,
    "HighChol": 0,
    "Age": 9,
    "Smoker": 0,
    "PhysActivity": 1,
    "Fruits": 1,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": 0,
    "DiffWalk": 0,
    "Sex": 1,
    "Education": 5,
    "Income": 7,
    "MentHlth": 2,
    "PhysHlth": 0,
    "CholCheck": 1,
    "Stroke": 0,
    "HeartDiseaseorAttack": 0
  }
}
```

> **Note:** Feature names and the set of expected keys will be driven by `feature_config.json`. The above is illustrative for the CDC Diabetes dataset.

**Response Body (200 OK):**

```json
{
  "predictions": [
    {
      "model_name": "Logistic Regression",
      "model_key": "logistic_regression",
      "prediction": 0,
      "prediction_label": "No Diabetes",
      "confidence": 0.87,
      "inference_time_ms": 0.042
    },
    {
      "model_name": "K-Nearest Neighbors",
      "model_key": "knn",
      "prediction": 0,
      "prediction_label": "No Diabetes",
      "confidence": 0.80,
      "inference_time_ms": 14.231
    },
    {
      "model_name": "Support Vector Machine",
      "model_key": "svm",
      "prediction": 1,
      "prediction_label": "Diabetes/Prediabetes",
      "confidence": 0.52,
      "inference_time_ms": 1.876
    },
    {
      "model_name": "Decision Tree",
      "model_key": "decision_tree",
      "prediction": 0,
      "prediction_label": "No Diabetes",
      "confidence": 0.91,
      "inference_time_ms": 0.018
    },
    {
      "model_name": "Naive Bayes",
      "model_key": "naive_bayes",
      "prediction": 0,
      "prediction_label": "No Diabetes",
      "confidence": 0.73,
      "inference_time_ms": 0.011
    }
  ],
  "consensus": {
    "majority_prediction": 0,
    "majority_label": "No Diabetes",
    "agreement_ratio": 0.80,
    "models_agreeing": 4,
    "models_total": 5
  },
  "input_echo": {
    "BMI": 28.5,
    "GenHlth": 3
  }
}
```

**Response Fields Explained:**

| Field | Type | Description |
|-------|------|-------------|
| `predictions` | `array` | One entry per model, always exactly 5 entries, in consistent order. |
| `predictions[].model_name` | `string` | Human-readable model name. |
| `predictions[].model_key` | `string` | Machine-readable model identifier (matches keys in `metrics.json`). |
| `predictions[].prediction` | `int` | Raw class prediction (0 or 1 for binary). |
| `predictions[].prediction_label` | `string` | Human-readable label for the predicted class. |
| `predictions[].confidence` | `float` | Maximum predicted probability (from `predict_proba`), 0.0–1.0. |
| `predictions[].inference_time_ms` | `float` | Wall-clock time for this model's `predict()` call, in milliseconds. |
| `consensus` | `object` | Summary of model agreement. |
| `consensus.majority_prediction` | `int` | The class predicted by the majority of models. |
| `consensus.majority_label` | `string` | Human-readable label for the majority prediction. |
| `consensus.agreement_ratio` | `float` | Fraction of models agreeing with the majority (e.g., 4/5 = 0.8). |
| `input_echo` | `object` | Echo of the submitted features (useful for debugging/display). |

**Error Responses:**

| Status | When | Response Body |
|--------|------|--------------|
| `422` | Missing or invalid feature values | `{ "detail": [ { "loc": ["body", "features", "BMI"], "msg": "field required", "type": "value_error.missing" } ] }` |
| `400` | Feature value out of valid range | `{ "error": "validation_error", "message": "BMI must be between 12.0 and 98.0, got 150.0", "field": "BMI" }` |
| `500` | Model failed to load or predict | `{ "error": "prediction_error", "message": "Model 'svm' failed: [error details]", "failed_models": ["svm"], "successful_predictions": [ ... ] }` |

> **Partial failure handling:** If one model fails, the other four predictions are still returned. The response includes both `successful_predictions` and `failed_models` so the frontend can display partial results gracefully.

---

#### `GET /api/v1/metrics`

Retrieve evaluation metrics for all five models.

**Response Body (200 OK):**

```json
{
  "generated_at": "2026-09-19T00:00:00Z",
  "dataset_info": {
    "name": "CDC Diabetes Health Indicators",
    "total_rows": 25000,
    "train_rows": 20000,
    "test_rows": 5000,
    "features_original": 22,
    "features_after_encoding": 42
  },
  "models": [
    {
      "model_name": "Logistic Regression",
      "model_key": "logistic_regression",
      "accuracy": 0.862,
      "precision": 0.841,
      "recall": 0.862,
      "f1_score": 0.847,
      "avg_inference_time_ms": 0.03,
      "model_file_size_kb": 4.2,
      "training_time_seconds": 1.7
    }
  ],
  "best_per_metric": {
    "accuracy": "logistic_regression",
    "precision": "svm",
    "recall": "logistic_regression",
    "f1_score": "svm",
    "fastest_inference": "naive_bayes",
    "smallest_model": "naive_bayes"
  }
}
```

---

#### `GET /api/v1/form-schema`

Returns feature metadata so the frontend can dynamically build the input form without hardcoding feature names.

**Response Body (200 OK):**

```json
{
  "features": [
    {
      "name": "BMI",
      "display_name": "Body Mass Index",
      "type": "numerical",
      "input_type": "number",
      "min": 12.0,
      "max": 98.0,
      "step": 0.1,
      "default": 28.0,
      "description": "Body Mass Index calculated from height and weight"
    },
    {
      "name": "GenHlth",
      "display_name": "General Health",
      "type": "categorical",
      "input_type": "select",
      "options": [
        { "value": 1, "label": "Excellent" },
        { "value": 2, "label": "Very Good" },
        { "value": 3, "label": "Good" },
        { "value": 4, "label": "Fair" },
        { "value": 5, "label": "Poor" }
      ],
      "default": 3,
      "description": "Would you say that in general your health is..."
    }
  ],
  "target_info": {
    "name": "Diabetes_binary",
    "classes": [
      { "value": 0, "label": "No Diabetes" },
      { "value": 1, "label": "Diabetes/Prediabetes" }
    ]
  }
}
```

---

#### `GET /api/v1/health`

Simple health check.

**Response Body (200 OK):**

```json
{
  "status": "healthy",
  "models_loaded": 5,
  "models": ["logistic_regression", "knn", "svm", "decision_tree", "naive_bayes"],
  "transformers_loaded": true,
  "uptime_seconds": 3621.4
}
```

**Response Body (503 Service Unavailable):**

```json
{
  "status": "unhealthy",
  "models_loaded": 3,
  "models_failed": ["svm", "knn"],
  "error": "Failed to load 2 model(s). Check server logs."
}
```

---

## 10. Frontend Requirements

The frontend is a single-page application (SPA) or a simple multi-page static site served alongside the FastAPI backend. **This is the one component where AI agents may generate code directly** — but still only when explicitly asked by Rahul.

### 10.1 Technology Choice

- **Default:** Vanilla HTML + CSS + JavaScript (no build tools, no npm).
- **Alternative (if Rahul decides):** React, Vue, or Svelte with a minimal build setup.
- The API contract in Section 9 is technology-agnostic — the frontend is a consumer of JSON APIs.

### 10.2 Pages

#### Page 1: Prediction Page (`/` or `/predict`)

**Purpose:** Let the user submit a data point and see all five models' predictions side by side.

**Layout & Components:**

| Component | Description |
|-----------|-------------|
| **Header** | "Algorithm Arena" title + navigation to Report Card page. |
| **Input Form** | Dynamically built from `GET /form-schema` response. Each feature gets an appropriate input element (number input, dropdown, slider). "Predict" button triggers `POST /predict`. |
| **Results Panel** | Appears after form submission. Shows 5 cards (one per model), each containing: model name, predicted class (with color coding: green = No Diabetes, red = Diabetes), confidence percentage (with a visual bar/gauge), inference latency in ms. |
| **Consensus Bar** | A summary strip showing: "4 out of 5 models predict No Diabetes (80% agreement)". Color-coded by agreement level. |
| **Loading State** | Spinner or skeleton cards while waiting for API response. |
| **Error State** | User-friendly error messages for validation errors or server failures. |

**UX Requirements:**

- Form should have sensible defaults (median values from `feature_config.json`).
- Results should be sortable by confidence or latency (click column header).
- The fastest and most confident model should be visually highlighted.
- If models disagree, visually emphasize the disagreement (e.g., a warning banner).

#### Page 2: Report Card Page (`/report`)

**Purpose:** Compare all five models on evaluation metrics. This is the "trade-off explorer."

**Layout & Components:**

| Component | Description |
|-----------|-------------|
| **Header** | Same header as Prediction page, navigation to Prediction page. |
| **Metrics Table** | A responsive table with: Rows = 5 models, Columns = Accuracy, Precision, Recall, F1 Score, Avg. Inference Time (ms), Model Size (KB). Best value per column is highlighted (bold + green background). Worst value per column is dimmed or marked (red text). |
| **Bar Charts** | Grouped bar chart showing all 5 models across the 4 accuracy metrics. Separate chart for inference time (log scale may be needed given the spread). Separate chart for model size. |
| **Trade-Off Narrative Callouts** | 2-3 insight cards auto-generated from the data, e.g.: "🏆 Naive Bayes is 100x faster than KNN but 5% less accurate.", "📦 KNN's model file is 3,600x larger than Logistic Regression because it stores the entire training set.", "⚖️ SVM achieves the best F1 score but takes 45 seconds to train — 900x longer than Naive Bayes." |
| **Dataset Info** | Sidebar or footer showing: dataset name, total rows, train/test split, number of features, date metrics were generated. |

**UX Requirements:**

- Metrics should be formatted to 3 decimal places for accuracy metrics, 2 decimal places for latency, whole numbers for model size.
- Charts should have tooltips on hover showing exact values.
- The table should be sortable by any column.
- The page should look good in a screenshot (LinkedIn-presentable).

#### Page 3: About Page (`/about`) — Optional / Nice-to-Have

**Purpose:** Explain the project, the algorithms, and the author.

| Component | Description |
|-----------|-------------|
| **Project Description** | What Algorithm Arena is and why it was built. |
| **Algorithm Summaries** | One paragraph per algorithm explaining what it does in plain English. |
| **Author Info** | Rahul's name, links to GitHub/LinkedIn. |

### 10.3 Design Guidelines

| Aspect | Guideline |
|--------|-----------|
| **Color Scheme** | Dark or light theme, professional. Suggest: dark background (#1a1a2e or similar) with accent colors per model (consistent across both pages). |
| **Typography** | Clean sans-serif (Inter, system-ui). Monospace for numbers/metrics. |
| **Responsiveness** | Desktop-first. Should not break on tablet. Mobile is nice-to-have. |
| **Accessibility** | Minimum: proper label-input associations, sufficient color contrast, keyboard-navigable form. |
| **Animation** | Subtle: card entrance animations when results load, chart transitions. Nothing distracting. |
| **Branding** | "Algorithm Arena" as the product name. Optionally a simple logo or icon. |

### 10.4 Frontend–Backend Integration

- The frontend makes `fetch()` calls to the backend API.
- In development, the frontend is served from the FastAPI backend via `StaticFiles` mount.
- CORS is configured to allow `localhost` origins during development.
- No authentication headers are needed.

---

## 11. Milestones / Build Order

Each milestone has explicit "done" criteria. These criteria are what Hermes Agent checks against when deciding what to write into `progress.md`.

### Milestone 1: Data Pipeline

**Goal:** Raw CSV → cleaned, encoded, scaled, split data + serialized transformers.

| Task | Done Criteria |
|------|--------------|
| 1.1 Download and inspect dataset | Raw CSV exists in `model_training/data/raw/`. EDA notebook exists in `model_training/notebooks/01_eda.ipynb` with shape, dtypes, distribution plots, missing-value analysis. |
| 1.2 Implement `preprocessing.py` | Script runs without errors. Produces: `scaler.pkl`, `encoder.pkl`, `feature_config.json` in `model_training/artifacts/`. Handles missing values, encoding, scaling as per Section 7. |
| 1.3 Verify no data leakage | Scaler and encoder are fitted on train set ONLY. Test set is transformed, not fitted. Verified by code review or assertion. |
| 1.4 Commit and update memory bank | Code is committed to git. Hermes updates `progress.md` to mark M1 complete. |

**Hermes checkpoint:** After Rahul confirms M1 is done → update `progress.md` (mark M1 ✅) and `activeContext.md` (shift focus to M2).

---

### Milestone 2: Model Training & Serialization

**Goal:** All 5 models trained, evaluated, serialized, metrics saved.

| Task | Done Criteria |
|------|--------------|
| 2.1 Implement `train.py` | Script trains all 5 models using processed data from M1. Each model's `fit()` runs without errors. |
| 2.2 Evaluation | `metrics.json` exists in `model_training/artifacts/` with all 7 metrics per model (accuracy, precision, recall, F1, avg inference time, model file size, training time). |
| 2.3 Serialization | Five `.pkl` model files exist in `model_training/artifacts/`. Each can be loaded via `joblib.load()` and used for `.predict()` and `.predict_proba()` without errors. |
| 2.4 Sanity check | A quick test: load each model + transformers, transform a sample row, predict with each model, verify output shape and type. Can be a simple script or notebook cell. |
| 2.5 Commit and update memory bank | Code is committed. Hermes updates `progress.md` and `activeContext.md`. |

**Hermes checkpoint:** After Rahul confirms M2 is done → update memory bank.

---

### Milestone 3: Backend API

**Goal:** FastAPI server serves predictions and metrics.

| Task | Done Criteria |
|------|--------------|
| 3.1 Project setup | `backend/main.py` creates a FastAPI app. `requirements.txt` has all dependencies pinned. Server starts with `uvicorn backend.main:app`. |
| 3.2 Model loading at startup | All 5 models + scaler + encoder + `feature_config.json` + `metrics.json` loaded into memory on app startup. Health endpoint returns `models_loaded: 5`. |
| 3.3 `POST /api/v1/predict` | Endpoint works as specified in Section 9. Accepts feature dict, transforms input, runs all 5 models, returns predictions with confidence and latency. Handles partial failures gracefully. |
| 3.4 `GET /api/v1/metrics` | Endpoint returns contents of `metrics.json` in the format specified in Section 9. |
| 3.5 `GET /api/v1/form-schema` | Endpoint returns feature metadata from `feature_config.json` in the format specified in Section 9. |
| 3.6 `GET /api/v1/health` | Endpoint returns health status. |
| 3.7 Input validation | Invalid/missing features return 422 or 400 with clear error messages. Out-of-range values are rejected with the expected range in the error message. |
| 3.8 CORS | CORS configured for local development. |
| 3.9 Commit and update memory bank | Code is committed. Hermes updates `progress.md` and `activeContext.md`. |

**Hermes checkpoint:** After Rahul confirms M3 is done → update memory bank.

---

### Milestone 4: Frontend

**Goal:** Working UI with Prediction page and Report Card page.

| Task | Done Criteria |
|------|--------------|
| 4.1 Prediction page | Form dynamically built from `/form-schema`. Submitting the form calls `/predict` and displays all 5 model results in cards. Loading and error states work. |
| 4.2 Report Card page | Metrics table populated from `/metrics`. All metrics displayed and formatted. Best/worst values highlighted. |
| 4.3 Charts | At least one bar chart comparing models on accuracy metrics. At least one chart for inference time. |
| 4.4 Trade-off callouts | 2-3 auto-generated insight sentences based on metric values. |
| 4.5 Navigation | User can navigate between Prediction and Report Card pages. |
| 4.6 Visual polish | Looks professional enough for a LinkedIn screenshot. Responsive on desktop. |
| 4.7 Commit and update memory bank | Code is committed. Hermes updates `progress.md` and `activeContext.md`. |

**Hermes checkpoint:** After Rahul confirms M4 is done → update memory bank.

---

### Milestone 5: Deployment

**Goal:** App is live on the internet, accessible via a public URL.

| Task | Done Criteria |
|------|--------------|
| 5.1 Deployment config | Dockerfile or platform-specific config (Render, Railway, etc.) exists. Environment variables documented. |
| 5.2 Deploy | App is accessible at a public URL. Prediction and Report Card pages work. |
| 5.3 Test deployed version | Send at least 3 test predictions to the live API. Verify results match local. |
| 5.4 Commit and update memory bank | Deployment configs committed. Hermes updates `progress.md` and `activeContext.md`. |

**Hermes checkpoint:** After Rahul confirms M5 is done → update memory bank.

---

### Milestone 6: LinkedIn Write-Up Assets

**Goal:** Create materials for the LinkedIn post.

| Task | Done Criteria |
|------|--------------|
| 6.1 Screenshots | At least 2 screenshots: Prediction results page showing model disagreement, Report Card page showing metrics comparison. |
| 6.2 README polish | `README.md` has: project description, tech stack, live demo link, architecture diagram, setup instructions, screenshots. |
| 6.3 LinkedIn post draft | A draft post (~200 words) focusing on the trade-off narrative, not the algorithms. |
| 6.4 Final memory bank update | Hermes updates `progress.md` to mark project COMPLETE. |

**Hermes checkpoint:** After Rahul confirms M6 is done → final memory bank update.

---

## 12. Edge Cases & Failure Modes

### 12.1 Input Validation Edge Cases

| Edge Case | Expected Behavior |
|-----------|-------------------|
| **Missing feature in request** | Return 422 with field name and message: "Missing required feature: {name}". |
| **Extra/unknown feature in request** | Ignore unknown fields silently (FastAPI's default with `model_config = ConfigDict(extra='ignore')`). Alternatively, return 400 warning. Document choice in `decisions.md`. |
| **Feature value out of range** (e.g., BMI = 500) | Return 400 with the valid range from `feature_config.json`: "BMI must be between 12.0 and 98.0, got 500.0". |
| **Feature value of wrong type** (e.g., BMI = "abc") | Return 422 via Pydantic type validation: "value is not a valid float". |
| **Negative values for inherently non-negative features** (e.g., Age = -5) | Return 400 with field-specific validation: "Age must be >= 0". |
| **All features at extreme minimum** | Process normally. Models may produce low-confidence predictions. This is valid behavior, not an error. |
| **All features at extreme maximum** | Same as above. Process normally. |
| **Empty request body** | Return 422: "field required: features". |
| **Request body is not valid JSON** | Return 422 (FastAPI handles this automatically). |
| **Integer sent for float field** | Accept silently (1 → 1.0). Python handles this natively. |
| **Float sent for integer/categorical field** | Round to nearest integer and process, OR reject with 400. Document choice. Recommended: accept and round. |

### 12.2 Model/System Failure Modes

| Failure Mode | Expected Behavior |
|-------------|-------------------|
| **One model .pkl file missing or corrupt** | Log error at startup. Health endpoint reports which models failed. `/predict` returns results for loaded models only, with a `failed_models` array in response. |
| **All model files missing** | Server starts but health returns 503. `/predict` returns 503 with message: "No models available". |
| **Scaler or encoder .pkl missing** | Server fails to start (fatal error). Log a clear error message pointing to the missing file and instructions to run `train.py`. |
| **`metrics.json` missing** | `/metrics` returns 404 with message: "Metrics file not found. Run train.py first." All other endpoints still work. |
| **`feature_config.json` missing** | Server fails to start (fatal — input validation depends on this file). |
| **Model prediction raises exception** (e.g., NaN in input after transformation) | Catch exception per-model. Return results for other models. Log the full traceback. |
| **Server runs out of memory** (unlikely at this scale) | Not handled — out of scope. Document that 5 sklearn models should fit in <500MB RAM total. |
| **Concurrent requests** | FastAPI handles concurrency via async. However, sklearn `predict()` is synchronous and holds the GIL. For this portfolio project, single-user latency is fine. Not a production concern. |

### 12.3 Data Pipeline Edge Cases

| Edge Case | Expected Behavior |
|-----------|-------------------|
| **Raw CSV has changed format** (new columns, renamed columns) | `preprocessing.py` should fail fast with a clear error listing expected vs. found columns. |
| **100% missing values in a column** | Drop the column. Log a warning. |
| **Target variable has >5 classes** | Log a warning. Proceed — all 5 algorithms support multi-class, but this was designed for 2-5 classes. |
| **Dataset is too small (<100 rows)** | Log a warning: "Dataset has only {n} rows. Results may not be statistically meaningful." Proceed anyway. |
| **Class with only 1 sample** | Stratified split will fail. Handle by either dropping the class or falling back to non-stratified split with a warning. |

---

## 13. Deployment Plan

### 13.1 Target Platforms (in preference order)

| Platform | Why | Cost |
|----------|-----|------|
| **Render** (recommended) | Free tier for web services. Easy FastAPI deployment. Auto-deploy from GitHub. | Free (with cold starts). |
| **Railway** | Simple deployment. Slightly more generous free tier. | Free tier with usage limits. |
| **Fly.io** | Good free tier. More control over regions/scaling. | Free for small apps. |
| **Heroku** | Classic, but free tier was removed in 2022. | Paid only (~$7/month minimum). |

> **Recommended: Render.** Deploy the FastAPI app as a "Web Service" from GitHub. Serve the frontend as static files mounted by FastAPI.

### 13.2 Deployment Architecture

```
GitHub Repo
    │
    ▼
Render Web Service (or equivalent)
    │
    ├── FastAPI app (backend/main.py)
    │   ├── Serves API endpoints (/api/v1/*)
    │   ├── Serves static frontend (frontend/*)
    │   └── Loads model artifacts from model_training/artifacts/
    │
    └── Model artifacts are COMMITTED to the repo
        (They're small enough — sklearn models for tabular data
         are typically <50MB total, well within git limits)
```

> **⚠️ Decision: Commit model artifacts to git or use cloud storage?**
> - For a portfolio project with sklearn models on tabular data, the total artifact size should be <50MB. Committing to git is simpler.
> - If KNN's model file is large (stores entire training set), consider: (a) reducing training set size for KNN, (b) using git LFS, or (c) adding a startup script that downloads from Google Drive/S3.
> - **Default recommendation: Commit to git.** Revisit if total artifact size exceeds 50MB. Document decision in `decisions.md`.

### 13.3 Environment Variables

| Variable | Purpose | Default | Required |
|----------|---------|---------|----------|
| `ARTIFACTS_PATH` | Path to the artifacts directory | `model_training/artifacts` | No (has default) |
| `HOST` | Server bind address | `0.0.0.0` | No |
| `PORT` | Server bind port | `8000` | Yes (Render sets this) |
| `CORS_ORIGINS` | Comma-separated allowed origins | `http://localhost:3000,http://localhost:8000` | No |
| `LOG_LEVEL` | Logging verbosity | `info` | No |

### 13.4 `.gitignore`

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
*.egg
.eggs/
venv/
.venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Data (raw data may be large — keep processed/artifacts tracked)
model_training/data/raw/*.csv
model_training/data/raw/*.zip
model_training/data/raw/*.gz

# Environment
.env
.env.local

# Logs
*.log

# Note: model_training/artifacts/ is NOT gitignored.
# The serialized models and metrics are committed to the repo
# for deployment simplicity. See Section 13.2 for rationale.
```

### 13.5 `requirements.txt`

```
# Core ML
scikit-learn>=1.3.0,<2.0.0
numpy>=1.24.0,<2.0.0
pandas>=2.0.0,<3.0.0
joblib>=1.3.0

# Backend
fastapi>=0.100.0,<1.0.0
uvicorn[standard]>=0.23.0,<1.0.0
pydantic>=2.0.0,<3.0.0

# Utilities
python-multipart>=0.0.6

# Development (optional, not needed in production)
# jupyter>=1.0.0
# matplotlib>=3.7.0
# seaborn>=0.12.0
```

### 13.6 Startup Command

```bash
# Local development
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Production (Render start command)
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

### 13.7 Render-Specific Configuration

If using Render:

| Setting | Value |
|---------|-------|
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| **Root Directory** | (leave blank — repo root) |
| **Python Version** | `3.11` (or latest stable) |
| **Plan** | Free |
| **Auto-Deploy** | Yes (from `main` branch) |

---

## 14. Open Questions

These are genuinely ambiguous items that affect scope or implementation. Each is flagged for Rahul's decision. Agents should NOT silently resolve these — they should check `decisions.md` for resolutions before proceeding.

| # | Question | Options | Impact | Default (if no answer) |
|---|----------|---------|--------|------------------------|
| OQ1 | **Dataset confirmation** ✅ RESOLVED | Online Shoppers Purchasing Intention (UCI ID 468). | — | — |
| OQ2 | **Encoding strategy** | Option A: OneHotEncoder for all models (simpler, fairer). Option B: OrdinalEncoder for trees, OneHotEncoder for others (more nuanced). | Affects pipeline complexity and number of artifacts. | Option A: OneHotEncoder for all. |
| OQ3 | **Frontend technology** | Vanilla HTML/CSS/JS (no build tools) or React/Vue/Svelte? | Affects folder structure, build process, agent scaffolding approach. | Vanilla HTML/CSS/JS. |
| OQ4 | **Subsampling strategy for large datasets** | Random subsample? Stratified subsample? Use full dataset? | Affects training time (especially SVM) and metric reliability. | Stratified subsample to ~25,000 rows. |
| OQ5 | **Naive Bayes variant** | GaussianNB (default), BernoulliNB (for binary features), CategoricalNB (for categorical features)? | Affects model behavior and potentially preprocessing. | GaussianNB (simplest, works with any continuous input). |
| OQ6 | **Extra features: handle or reject?** | Silently ignore extra features in the request body, or return 400? | API behavior for unexpected input. | Silently ignore (FastAPI default). |
| OQ7 | **Float-to-int rounding for categorical fields** | Accept and round, or reject with 400? | API strictness level. | Accept and round. |
| OQ8 | **Commit artifacts to git or external storage?** | Git (simple), Git LFS (if large), S3/GDrive (complex). | Deployment complexity. | Commit to git. Revisit if >50MB. |
| OQ9 | **Deployment platform** | Render (recommended), Railway, Fly.io, other? | Affects deployment config, startup commands. | Render. |
| OQ10 | **LinkedIn post — single post or article?** | Short post with screenshots, or long-form article? | Affects M6 scope. | Short post. Decide later. |

> **Instruction for agents:** When you encounter one of these questions while working, check `/memory-bank/decisions.md` first. If the question has been resolved there, use that answer. If not, ask Rahul — do not guess.

---

## 15. Glossary / Context for Future Agents

This section is written for an AI agent that has **never seen this project before** and is starting a fresh session with no memory of any prior conversation. Read this section first.

### 15.1 Project-Specific Terms

| Term | Definition |
|------|-----------|
| **Algorithm Arena** | The name of this project. A web app that compares 5 classical ML models on the same dataset. |
| **Report Card** | The page/view that shows evaluation metrics (accuracy, precision, recall, F1, latency, model size) for all 5 models side by side. The term is used throughout the PRD and frontend. |
| **Trade-off narrative** | The central thesis of the project: no single model is best on every dimension. One model is fastest, another is most accurate, another is smallest. Algorithm Arena makes these trade-offs visible. |
| **Artifacts (model artifacts)** | The serialized `.pkl` files for models and transformers, plus `metrics.json` and `feature_config.json`, stored in `model_training/artifacts/`. These are the output of offline training and the input to online serving. |
| **Memory Bank** | The `/memory-bank/` directory at the repo root. Contains 4 markdown files that track project state across sessions. NOT part of the application runtime — purely project metadata for agent handoff. |
| **Hermes Agent** | A specific AI agent (separate from coding agents) whose sole job is maintaining the memory bank files. It does not write application code. It reads project state when told to and updates `activeContext.md` and `progress.md`. |
| **Offline Training Pipeline** | The scripts (`preprocessing.py`, `train.py`) that run locally on Rahul's machine to produce model artifacts. Not part of the deployed web app. |
| **Online Serving App** | The FastAPI backend + frontend that serves predictions using pre-trained models. This is what gets deployed. |
| **Feature Config** | `feature_config.json` — a JSON file listing all features, their types, valid ranges, and categories. Used by both the backend (input validation) and frontend (form generation). |
| **Consensus** | In the `/predict` response, the consensus object shows how many models agree on the prediction. A disagreement (e.g., 3-2 split) is interesting and highlights trade-offs. |

### 15.2 Tech Stack Summary

| Layer | Technology | Version |
|-------|-----------|---------|
| ML Training | Python, scikit-learn, pandas, numpy | Python 3.11+, sklearn ≥1.3 |
| Backend | FastAPI, Uvicorn, Pydantic v2 | FastAPI ≥0.100 |
| Frontend | Vanilla HTML/CSS/JS (unless changed — check `decisions.md`) | N/A |
| Serialization | joblib (for sklearn models) | ≥1.3 |
| Deployment | Render (unless changed — check `decisions.md`) | N/A |

### 15.3 Division of Labor (Restated for Cold-Start Agents)

**READ THIS BEFORE DOING ANYTHING:**

1. **Rahul writes all backend, ML pipeline, and API code.** Do not generate this code unless Rahul explicitly asks you to help with a specific piece.
2. **Frontend code is the ONE exception** — agents may generate frontend code, but ONLY when Rahul asks.
3. **Hermes Agent maintains the memory bank.** It does not write application code. If you are not Hermes, do not edit memory bank files.
4. **Any agent may assist with debugging and code review** — but only when asked.
5. **Before starting any work in a new session,** read these files in order:
   - `/memory-bank/activeContext.md` — what's currently being worked on.
   - `/memory-bank/progress.md` — what's done, what's in progress, what's next.
   - `/memory-bank/decisions.md` — resolved design questions.
   - `PRD.md` (this file) — the full specification.

**If you skip the memory bank files, you risk duplicating work, contradicting decisions, or working on the wrong milestone.**

---

## 16. Cross-Session Memory & Agent Handoff Protocol

This section is the most critical section for maintaining project continuity across sessions, tools, and models. Read it carefully.

### 16.1 Memory Bank Files

The `/memory-bank/` directory at the repository root contains exactly four files:

| File | Purpose | Who Updates | Update Frequency |
|------|---------|-------------|-----------------|
| `projectbrief.md` | Mirrors the core facts from `PRD.md` (project name, goals, tech stack, architecture). Provides a quick-reference summary so agents don't have to read the full PRD for basic context. | Rahul (manually, if PRD changes significantly). | Rarely — only when fundamental project scope changes. |
| `activeContext.md` | Describes what is currently being worked on, what the immediate next task is, any blockers, and any temporary context (e.g., "SVM training is slow, considering subsampling — see OQ4"). | **Hermes Agent** (when told to by Rahul). | After each meaningful work session or when focus shifts between milestones. |
| `progress.md` | Running log of completed milestones, in-progress tasks, and next tasks. Uses checkboxes: ✅ Done, 🔄 In-progress, 📋 Next. References milestone numbers from Section 11. | **Hermes Agent** (when told to by Rahul). | After each milestone completion and during milestones when sub-tasks complete. |
| `decisions.md` | Records key architectural and design decisions with their rationale. Format: Question → Decision → Rationale → Date. Prevents re-litigation of resolved questions. | **Hermes Agent** (when told to by Rahul). Rahul may also edit directly. | Whenever a decision from the Open Questions list (Section 14) is resolved, or a new architectural choice is made. |

### 16.2 Hermes Agent's Standing Role

**Hermes Agent is the project's memory keeper.** Its job is:

1. **READ** the current state of the project when asked. This means Hermes reads:
   - The files in `/memory-bank/` (its own prior outputs).
   - Key project files (especially `model_training/artifacts/` to check what exists).
   - Git history (`git log --oneline -20` or similar) if available.
   - Any direct context Rahul provides in the prompt.

2. **WRITE/UPDATE** the memory bank files based on what it reads:
   - Update `activeContext.md` to reflect the current focus.
   - Update `progress.md` to mark tasks as done/in-progress/next.
   - Update `decisions.md` when Rahul communicates a resolved decision.
   - Rarely, update `projectbrief.md` if the project scope changes.

3. **NOT DO** the following:
   - Write application code (backend, ML, frontend, API — none of it).
   - Make architectural decisions autonomously.
   - Monitor the repo in real time (it has no passive awareness).
   - Update memory bank files unless explicitly told to by Rahul.

**Hermes has no automatic visibility into what other agents or Rahul do.** It only knows what it's told. This is by design — Rahul controls the information flow.

### 16.3 Trigger Points for Memory Bank Updates

Rahul will prompt Hermes to update the memory bank at these specific moments:

| Trigger | What Rahul Tells Hermes | What Hermes Does |
|---------|------------------------|------------------|
| **Milestone completed** (M1 through M6) | "Update the memory bank — Milestone {N} is done." | Mark the milestone ✅ in `progress.md`. Move `activeContext.md` focus to the next milestone. |
| **Significant sub-task completed** | "Update the memory bank — I finished {specific task}." | Update `progress.md` with the sub-task. Optionally update `activeContext.md` if focus shifted. |
| **Architectural decision made** | "Update the memory bank — decided on {X} because {Y}." | Add entry to `decisions.md` with question, decision, rationale, and date. Update `activeContext.md` if the decision unblocks work. |
| **End of work session** | "Update the memory bank — done for today. Here's where I left off: {summary}." | Update `activeContext.md` with the stopping point, any blockers, and immediate next steps. Update `progress.md` with any completed items. |
| **Blocker encountered** | "Update the memory bank — blocked on {X}." | Add blocker to `activeContext.md` with details and any attempted solutions. |
| **Scope change** | "Update the memory bank — we're changing {X}." | Update relevant files. If it's a fundamental change, update `projectbrief.md`. |

### 16.4 Standing Instructions for Non-Hermes Agents

**Every coding agent (Claude, Gemini, Antigravity, Cursor, Copilot, or any other AI tool)** must follow these rules when starting a new session on this project. These rules are encoded in the project's rules files (`.antigravityrules`, `CLAUDE.md`, etc.):

#### Mandatory Pre-Work Checklist

```
□ Read /memory-bank/activeContext.md in full.
□ Read /memory-bank/progress.md in full.
□ Read /memory-bank/decisions.md in full.
□ Do NOT assume any prior conversational context exists.
□ Do NOT assume you know what has been done or decided — the memory bank is your source of truth.
□ If memory bank files don't exist yet, read PRD.md (this file) instead.
```

#### Behavioral Rules

1. **Never write backend, ML pipeline, or API code** unless Rahul explicitly asks for help with a specific piece.
2. **Frontend code may be generated** only when Rahul requests it.
3. **Never edit memory bank files** — that's Hermes's job.
4. **When encountering an Open Question** (Section 14), check `decisions.md` first. If unresolved, ask Rahul. Do not guess.
5. **After completing work,** summarize what was done and suggest that Rahul update the memory bank (remind Rahul to tell Hermes).
6. **Reference PRD section numbers** when discussing requirements (e.g., "Per PRD Section 9.2, the `/predict` endpoint should return...").

### 16.5 Manual vs. Automated Memory Updates

> **This protocol is a discipline that Rahul enforces manually through explicit prompts to Hermes.** It is NOT an automated pipeline. There is no cron job, git hook, or CI action that triggers memory bank updates.

Rahul must remember to:
- Tell Hermes to update after each milestone.
- Tell Hermes to update at the end of each work session.
- Tell Hermes about decisions as they're made.

If Rahul forgets, the memory bank becomes stale, and the next agent session may work from outdated context. **This is an accepted risk** of the manual approach, traded for simplicity.

**Future automation (nice-to-have, out of scope for MVP):**
- A git post-commit hook that appends to `progress.md`.
- A CI action that runs Hermes on every push.
- A scheduled job that has Hermes review the repo weekly.

These are explicitly out of scope for the initial build. Rahul may add them later.

### 16.6 Memory Bank Bootstrap

When the project is first created and the memory bank directory is empty, the following files should be initialized:

#### `projectbrief.md` — Initial Content

```markdown
# Algorithm Arena — Project Brief

## What It Is
A web app comparing 5 classical ML models (Logistic Regression, KNN, SVM, Decision Tree,
Naive Bayes) on the same dataset, showing trade-offs in accuracy, speed, and model size.

## Owner
Rahul (CSE B.Tech student)

## Tech Stack
- ML: Python, scikit-learn, pandas, numpy
- Backend: FastAPI, Uvicorn, Pydantic v2
- Frontend: Vanilla HTML/CSS/JS
- Deployment: Render (pending confirmation)

## Key Principle
The point is the TRADE-OFF NARRATIVE — no single model wins on every dimension.

## Full Spec
See PRD.md for complete requirements.
```

#### `activeContext.md` — Initial Content

```markdown
# Active Context

## Current Focus
Project setup. PRD is written. No code exists yet.

## Immediate Next Steps
1. Rahul to confirm dataset choice (see PRD Section 4, OQ1).
2. Begin Milestone 1: Data Pipeline.

## Blockers
None.

## Session Notes
- 2026-09-19: PRD created. Project initialized.
```

#### `progress.md` — Initial Content

```markdown
# Progress Log

## Milestones

- 📋 **M1: Data Pipeline** — Not started
  - [ ] 1.1 Download and inspect dataset
  - [ ] 1.2 Implement preprocessing.py
  - [ ] 1.3 Verify no data leakage
  - [ ] 1.4 Commit and update memory bank

- 📋 **M2: Model Training & Serialization** — Not started
  - [ ] 2.1 Implement train.py
  - [ ] 2.2 Evaluation
  - [ ] 2.3 Serialization
  - [ ] 2.4 Sanity check
  - [ ] 2.5 Commit and update memory bank

- 📋 **M3: Backend API** — Not started
  - [ ] 3.1 Project setup
  - [ ] 3.2 Model loading at startup
  - [ ] 3.3 POST /predict
  - [ ] 3.4 GET /metrics
  - [ ] 3.5 GET /form-schema
  - [ ] 3.6 GET /health
  - [ ] 3.7 Input validation
  - [ ] 3.8 CORS
  - [ ] 3.9 Commit and update memory bank

- 📋 **M4: Frontend** — Not started
  - [ ] 4.1 Prediction page
  - [ ] 4.2 Report Card page
  - [ ] 4.3 Charts
  - [ ] 4.4 Trade-off callouts
  - [ ] 4.5 Navigation
  - [ ] 4.6 Visual polish
  - [ ] 4.7 Commit and update memory bank

- 📋 **M5: Deployment** — Not started
  - [ ] 5.1 Deployment config
  - [ ] 5.2 Deploy
  - [ ] 5.3 Test deployed version
  - [ ] 5.4 Commit and update memory bank

- 📋 **M6: LinkedIn Write-Up Assets** — Not started
  - [ ] 6.1 Screenshots
  - [ ] 6.2 README polish
  - [ ] 6.3 LinkedIn post draft
  - [ ] 6.4 Final memory bank update
```

#### `decisions.md` — Initial Content

```markdown
# Decisions Log

## Format
Each entry: Question → Decision → Rationale → Date

---

_No decisions recorded yet. See PRD.md Section 14 for open questions pending resolution._
```

---

## Appendix A: Quick Reference Card

For agents that need to orient quickly without reading the full PRD:

```
PROJECT:        Algorithm Arena
OWNER:          Rahul
WHAT:           Web app comparing 5 ML models on same dataset
MODELS:         Logistic Regression, KNN, SVM, Decision Tree, Naive Bayes
DATASET:        CDC Diabetes Health Indicators (pending confirmation)
BACKEND:        FastAPI (Python 3.11+)
FRONTEND:       Vanilla HTML/CSS/JS
DEPLOYMENT:     Render (free tier)
SERIALIZATION:  joblib (.pkl files)
API PREFIX:     /api/v1
KEY ENDPOINTS:  POST /predict, GET /metrics, GET /form-schema, GET /health

CRITICAL RULES:
1. Rahul writes all backend/ML/API code.
2. Agents generate frontend code ONLY when asked.
3. Hermes Agent maintains /memory-bank/ — nobody else edits those files.
4. Read /memory-bank/ files BEFORE doing anything in a new session.
5. Never refit transformers at inference time.
6. Check decisions.md before guessing at open questions.
```

---

## Appendix B: Dependency Map

```
preprocessing.py
    ├── Produces: scaler.pkl, encoder.pkl, feature_config.json
    ├── Produces: X_train, X_test, y_train, y_test (in-memory or .npy)
    └── Depends on: raw CSV in data/raw/

train.py
    ├── Produces: 5 × model.pkl, metrics.json
    ├── Depends on: preprocessing.py output
    └── Depends on: scaler.pkl, encoder.pkl (for latency testing with transform)

backend/main.py
    ├── Depends on: all .pkl files in artifacts/
    ├── Depends on: metrics.json
    ├── Depends on: feature_config.json
    └── Does NOT depend on: preprocessing.py or train.py at runtime

frontend/*
    ├── Depends on: API endpoints (GET /form-schema, POST /predict, GET /metrics)
    └── Does NOT depend on: any Python code directly
```

---

*End of PRD.md — Document version 1.0.0*
