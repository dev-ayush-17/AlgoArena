# 00 — Workflow & Git Map — Algorithm Arena (ML Track)

Welcome to the Algorithm Arena Machine Learning Guide. This document serves as the primary entry point, index, and git workflow map for Milestone 1 (Data Pipeline) and Milestone 2 (Model Training & Serialization).

---

## 1. Branching Strategy & Git Protocol

To ensure clean isolation of features and prevent broken code from polluting the stable `main` branch, we enforce a strict multi-tier branch workflow:

### Long-Lived Branches
* `main`: Protected, stable production branch. Never commit directly to `main`.
* `ml`: Long-lived integration branch for all ML pipeline and model training work.
* `backend`: Long-lived branch for FastAPI server (Milestone 3).
* `frontend`: Long-lived branch for web UI (Milestone 4).

### Short-Lived Feature Branches (ML Track)
Every task within the ML track must be built on a dedicated feature branch created off `ml`. Once the feature code is completed and verified against its corresponding guide document, submit a PR to merge back into `ml`.

| Feature Branch | Purpose | Guide File Reference |
| :--- | :--- | :--- |
| `ml/feature-eda` | Dataset download, inspection, univariate/bivariate EDA, correlation analysis | `01-data-understanding-and-eda.md` |
| `ml/feature-preprocessing` | Missing values, encoding, scaling, train/test split, transformer serialization | `02-preprocessing-and-feature-engineering.md` |
| `ml/feature-logistic-regression` | Logistic Regression model training, evaluation, latency benchmark, export | `04-logistic-regression.md` |
| `ml/feature-knn` | K-Nearest Neighbors model training, evaluation, latency benchmark, export | `05-knn.md` |
| `ml/feature-svm` | Support Vector Machine model training, evaluation, latency benchmark, export | `06-svm.md` |
| `ml/feature-decision-tree` | Decision Tree model training, evaluation, latency benchmark, export | `07-decision-tree.md` |
| `ml/feature-naive-bayes` | Naive Bayes model training, evaluation, latency benchmark, export | `08-naive-bayes.md` |
| `ml/feature-comparison-and-serialization` | Metrics aggregation table, `metrics.json` export, artifact verification | `09-model-comparison-and-serialization.md` |

---

## 2. Commit Message Convention

All commits within the ML track must use standardized semantic commit message prefixes to maintain clear history across feature branches:

* `feat(eda): ...` — Exploratory data analysis scripts or notebooks
* `feat(preprocess): ...` — Data cleaning, encoding, scaling, or pipeline transformers
* `feat(model-logreg): ...` — Logistic Regression model training or serialization
* `feat(model-knn): ...` — KNN model training or serialization
* `feat(model-svm): ...` — SVM model training or serialization
* `feat(model-dt): ...` — Decision Tree model training or serialization
* `feat(model-nb): ...` — Naive Bayes model training or serialization
* `feat(model-eval): ...` — Comparative evaluation, metrics aggregation, or artifact validation

---

## 3. Guide Table of Contents & File Index

The documentation suite in `/docs/ml-guide/` is organized sequentially:

1. **`00-workflow-and-git-map.md`** *(This file)*: Git workflow, branch mapping, commit standards, and guide directory index.
2. **`01-data-understanding-and-eda.md`**: Loading the UCI dataset (ID 468), structure checks, class imbalance verification, feature distributions, correlation heatmaps.
3. **`02-preprocessing-and-feature-engineering.md`**: Categorical encoding (`Month`, `VisitorType`, `Weekend`), numeric scaling (`StandardScaler`), stratified train-test splitting, transformer serialization (`scaler.pkl`, `encoder.pkl`).
4. **`03-visualization-toolkit.md`**: Reusable plotting functions for Confusion Matrices, ROC Curves, Precision-Recall Curves, Feature Importance, Decision Boundaries, and comparative bar charts.
5. **`04-logistic-regression.md`**: Baseline linear classification model instantiation, hyperparameter tuning (`C`, `max_iter`), evaluation, inference latency benchmarking, `logistic_regression.pkl` serialization.
6. **`05-knn.md`**: Distance-based classification model instantiation (`n_neighbors`), scaling impact analysis, evaluation, inference latency benchmarking, `knn.pkl` serialization.
7. **`06-svm.md`**: Support Vector Classifier instantiation (`kernel='rbf'`, `probability=True`), boundary mapping, evaluation, latency benchmarking, `svm.pkl` serialization.
8. **`07-decision-tree.md`**: Non-linear tree classification model instantiation (`max_depth`), feature importance extraction, evaluation, latency benchmarking, `decision_tree.pkl` serialization.
9. **`08-naive-bayes.md`**: Probabilistic classification model instantiation (`GaussianNB`), independence assumption evaluation on correlated features (`BounceRates` vs `ExitRates`), evaluation, latency benchmarking, `naive_bayes.pkl` serialization.
10. **`09-model-comparison-and-serialization.md`**: Master evaluation table generation across all 5 models, JSON serialization (`metrics.json`), artifact readiness checklist before backend integration.
