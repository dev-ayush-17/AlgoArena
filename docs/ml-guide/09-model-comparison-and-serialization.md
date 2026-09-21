# 09 — Model Comparison & Artifact Serialization

This document guides you through consolidating test metrics from all 5 trained algorithms into a unified comparison table, exporting the canonical `metrics.json` file for backend consumption, and executing the final artifact verification checklist.

---

## Step 1: Metric Aggregation & Comparative Summary Table

### 1. What & Why
Consolidating evaluation results across all 5 models (Accuracy, Precision, Recall, F1-Score, Single-Row Inference Latency, Model File Size, and Training Time) provides an apples-to-apples trade-off matrix. On our imbalanced dataset, this table exposes why model selection is an engineering trade-off between speed, size, precision, and recall.

### 2. Code
```python
import pandas as pd

# Construct comparative benchmark summary DataFrame
metrics_summary = [
    {
        "model_key": "logistic_regression",
        "display_name": "Logistic Regression",
        "accuracy": round(acc_logreg, 4),
        "precision": round(prec_logreg, 4),
        "recall": round(rec_logreg, 4),
        "f1_score": round(f1_logreg, 4),
        "avg_inference_time_ms": round(avg_latency_logreg, 4),
        "model_file_size_kb": round(size_logreg_kb, 2),
        "training_time_seconds": round(train_logreg_sec, 4)
    },
    {
        "model_key": "knn",
        "display_name": "K-Nearest Neighbors",
        "accuracy": round(acc_knn, 4),
        "precision": round(prec_knn, 4),
        "recall": round(rec_knn, 4),
        "f1_score": round(f1_knn, 4),
        "avg_inference_time_ms": round(avg_latency_knn, 4),
        "model_file_size_kb": round(size_knn_kb, 2),
        "training_time_seconds": round(train_knn_sec, 4)
    },
    {
        "model_key": "svm",
        "display_name": "Support Vector Machine",
        "accuracy": round(acc_svm, 4),
        "precision": round(prec_svm, 4),
        "recall": round(rec_svm, 4),
        "f1_score": round(f1_svm, 4),
        "avg_inference_time_ms": round(avg_latency_svm, 4),
        "model_file_size_kb": round(size_svm_kb, 2),
        "training_time_seconds": round(train_svm_sec, 4)
    },
    {
        "model_key": "decision_tree",
        "display_name": "Decision Tree",
        "accuracy": round(acc_dt, 4),
        "precision": round(prec_dt, 4),
        "recall": round(rec_dt, 4),
        "f1_score": round(f1_dt, 4),
        "avg_inference_time_ms": round(avg_latency_dt, 4),
        "model_file_size_kb": round(size_dt_kb, 2),
        "training_time_seconds": round(train_dt_sec, 4)
    },
    {
        "model_key": "naive_bayes",
        "display_name": "Naive Bayes",
        "accuracy": round(acc_nb, 4),
        "precision": round(prec_nb, 4),
        "recall": round(rec_nb, 4),
        "f1_score": round(f1_nb, 4),
        "avg_inference_time_ms": round(avg_latency_nb, 4),
        "model_file_size_kb": round(size_nb_kb, 2),
        "training_time_seconds": round(train_nb_sec, 4)
    }
]

df_summary = pd.DataFrame(metrics_summary).set_index("display_name")
print("=== ALGORITHM ARENA BENCHMARK SUMMARY ===")
print(df_summary[['accuracy', 'precision', 'recall', 'f1_score', 'avg_inference_time_ms', 'model_file_size_kb']])
```

### 3. Expected Output & How to Read It
* Output: 5-row table comparing all models across 6 key metrics.
* Trade-off Insights:
  - **Fastest Inference:** Decision Tree / Naive Bayes (~0.02 ms).
  - **Smallest Model Size:** Naive Bayes / Logistic Regression (~3-5 KB).
  - **Largest Model Size / Highest Latency:** KNN (~2,000 KB file size, ~10 ms latency).
  - **Highest F1 / Balance:** Decision Tree / RBF SVM (~0.65 - 0.68 F1 on class 1).

### 4. Common Pitfalls
* Rounding metrics prematurely before performing JSON serialization formatting.
* Mixing up single-row latency with total test set prediction time.

### 5. Git Checkpoint
* **Branch:** `ml/feature-comparison-and-serialization`
* **Commit Message:** `feat(model-eval): generate comparative summary table across all 5 algorithms`

---

## Step 2: Exporting Production `metrics.json`

### 1. What & Why
The FastAPI backend `GET /api/v1/metrics` endpoint serves pre-computed benchmark metrics directly from `model_training/artifacts/metrics.json`. Writing this file in the exact JSON schema defined in PRD Section 8.3 connects offline training results directly to live online serving.

### 2. Code
```python
import os
import json
from datetime import datetime, timezone

artifacts_dir = os.path.join("model_training", "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)
metrics_json_path = os.path.join(artifacts_dir, "metrics.json")

# Build JSON structure strictly adhering to PRD API Contract
metrics_payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "dataset": "Online Shoppers Purchasing Intention Dataset (UCI ID 468)",
    "dataset_size": {
        "total_rows": 12330,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": 17,
        "features_after_encoding": X_train_processed.shape[1]
    },
    "models": {
        item["model_key"]: {
            "display_name": item["display_name"],
            "accuracy": item["accuracy"],
            "precision": item["precision"],
            "recall": item["recall"],
            "f1_score": item["f1_score"],
            "avg_inference_time_ms": item["avg_inference_time_ms"],
            "model_file_size_kb": item["model_file_size_kb"],
            "training_time_seconds": item["training_time_seconds"]
        } for item in metrics_summary
    }
}

with open(metrics_json_path, "w") as f:
    json.dump(metrics_payload, f, indent=2)

print(f"Successfully generated metrics.json at: {metrics_json_path}")
```

### 3. Expected Output & How to Read It
* File `metrics.json` written under `/model_training/artifacts/`.
* Validation: Opening `metrics.json` shows a valid JSON object containing timestamp, dataset metadata, and a dictionary keyed by `logistic_regression`, `knn`, `svm`, `decision_tree`, and `naive_bayes`.

### 4. Common Pitfalls
* Changing key names (e.g. using `f1` instead of `f1_score`), which causes Pydantic schema validation failures in FastAPI during backend startup.

### 5. Git Checkpoint
* **Branch:** `ml/feature-comparison-and-serialization`
* **Commit Message:** `feat(model-eval): serialize production metrics.json adhering to PRD API contract`

---

## Step 3: Exporting `feature_config.json`

### 1. What & Why
The frontend dynamic form generator (`GET /api/v1/form-schema`) requires feature metadata (feature names, data types, min/max numerical bounds, categorical options, and default values). Generating `feature_config.json` provides the authoritative schema for both frontend rendering and backend payload validation.

### 2. Code
```python
feature_config_path = os.path.join(artifacts_dir, "feature_config.json")

feature_config = {
    "numerical_features": [
        {"name": "Administrative", "dtype": "int64", "min": 0, "max": 27, "default": 0, "description": "Number of administrative pages visited"},
        {"name": "Administrative_Duration", "dtype": "float64", "min": 0.0, "max": 3398.0, "default": 0.0, "description": "Total time spent on administrative pages (seconds)"},
        {"name": "Informational", "dtype": "int64", "min": 0, "max": 24, "default": 0, "description": "Number of informational pages visited"},
        {"name": "Informational_Duration", "dtype": "float64", "min": 0.0, "max": 2549.0, "default": 0.0, "description": "Total time spent on informational pages (seconds)"},
        {"name": "ProductRelated", "dtype": "int64", "min": 0, "max": 705, "default": 1, "description": "Number of product-related pages visited"},
        {"name": "ProductRelated_Duration", "dtype": "float64", "min": 0.0, "max": 63973.5, "default": 10.0, "description": "Total time spent on product-related pages (seconds)"},
        {"name": "BounceRates", "dtype": "float64", "min": 0.0, "max": 0.2, "default": 0.0, "description": "Percentage of visitors entering site and leaving immediately"},
        {"name": "ExitRates", "dtype": "float64", "min": 0.0, "max": 0.2, "default": 0.02, "description": "Percentage of pageviews that were last in session"},
        {"name": "PageValues", "dtype": "float64", "min": 0.0, "max": 361.0, "default": 0.0, "description": "Average Google Analytics value of visited pages"},
        {"name": "SpecialDay", "dtype": "float64", "min": 0.0, "max": 1.0, "default": 0.0, "description": "Closeness of site visit time to a special day (0.0 to 1.0)"}
    ],
    "categorical_features": [
        {"name": "Month", "dtype": "string", "categories": ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], "default": "May", "description": "Month of the session"},
        {"name": "OperatingSystems", "dtype": "int64", "categories": [1, 2, 3, 4, 5, 6, 7, 8], "default": 1, "description": "Operating system ID"},
        {"name": "Browser", "dtype": "int64", "categories": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "default": 2, "description": "Browser ID"},
        {"name": "Region", "dtype": "int64", "categories": [1, 2, 3, 4, 5, 6, 7, 8, 9], "default": 1, "description": "Geographic region ID"},
        {"name": "TrafficType", "dtype": "int64", "categories": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "default": 2, "description": "Traffic source type ID"},
        {"name": "VisitorType", "dtype": "string", "categories": ["Returning_Visitor", "New_Visitor", "Other"], "default": "Returning_Visitor", "description": "Visitor status"},
        {"name": "Weekend", "dtype": "int64", "categories": [0, 1], "default": 0, "description": "Session occurred on weekend (0=No, 1=Yes)"}
    ],
    "target": {
        "name": "Revenue",
        "classes": [0, 1],
        "class_labels": ["No Purchase", "Purchase Completed"]
    }
}

with open(feature_config_path, "w") as f:
    json.dump(feature_config, f, indent=2)

print(f"Successfully generated feature_config.json at: {feature_config_path}")
```

### 3. Expected Output & How to Read It
* Output: `feature_config.json` containing complete UI form specification.

### 4. Common Pitfalls
* Omitting string options for `Month` or `VisitorType`, breaking frontend dropdown menu population.

### 5. Git Checkpoint
* **Branch:** `ml/feature-comparison-and-serialization`
* **Commit Message:** `feat(model-eval): export feature_config.json for frontend schema endpoint`

---

## Step 4: Final Milestone Artifact Verification Checklist

### 1. What & Why
Before declaring Milestone 1 and Milestone 2 complete and transitioning to Milestone 3 (FastAPI Backend), all 9 required artifact files must exist in `/model_training/artifacts/`.

### 2. Code
```python
import os

required_artifacts = [
    "logistic_regression.pkl",
    "knn.pkl",
    "svm.pkl",
    "decision_tree.pkl",
    "naive_bayes.pkl",
    "scaler.pkl",
    "encoder.pkl",
    "feature_config.json",
    "metrics.json"
]

artifacts_dir = os.path.join("model_training", "artifacts")
print("=== FINAL ARTIFACT CHECKLIST VERIFICATION ===")

missing_count = 0
for artifact in required_artifacts:
    path = os.path.join(artifacts_dir, artifact)
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024.0
        print(f"  [x] {artifact:<25} ({size_kb:>8.2f} KB)")
    else:
        print(f"  [ ] {artifact:<25} MISSING!")
        missing_count += 1

if missing_count == 0:
    print("\nALL 9 ARTIFACTS VERIFIED SUCCESSFULLY. Ready for Milestone 3 (Backend FastAPI)!")
else:
    print(f"\nWARNING: {missing_count} required artifact(s) missing. Resolve before proceeding.")
```

### 3. Expected Output & How to Read It
```
=== FINAL ARTIFACT CHECKLIST VERIFICATION ===
  [x] logistic_regression.pkl   (    4.20 KB)
  [x] knn.pkl                   ( 2450.12 KB)
  [x] svm.pkl                   (  850.40 KB)
  [x] decision_tree.pkl         (   14.80 KB)
  [x] naive_bayes.pkl           (    3.10 KB)
  [x] scaler.pkl                (    1.20 KB)
  [x] encoder.pkl               (    2.40 KB)
  [x] feature_config.json       (    1.80 KB)
  [x] metrics.json              (    2.10 KB)

ALL 9 ARTIFACTS VERIFIED SUCCESSFULLY. Ready for Milestone 3 (Backend FastAPI)!
```

### 4. Common Pitfalls
* Forgetting to verify pickle re-loadability before transitioning to backend construction.

### 5. Git Checkpoint
* **Branch:** `ml/feature-comparison-and-serialization`
* **Commit Message:** `feat(model-eval): complete final ML artifact validation checklist`
