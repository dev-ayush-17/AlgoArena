# 07 — Model 4: Decision Tree

This document covers training, evaluating, latency benchmarking, and serializing a **Decision Tree Classifier** on the preprocessed Online Shoppers dataset.

---

## Step 1: Model Instantiation & Hyperparameter Setup

### 1. What & Why
Decision Trees recursively partition feature space into axis-aligned hyperplanes by maximizing Information Gain or Gini impurity reduction. Unconstrained trees grow deep branches that overfit noise in minority classes. Setting `max_depth=6`, `min_samples_split=10`, and `min_samples_leaf=5` bounds tree growth, ensuring high generalization on test data while maintaining full model interpretability. `class_weight='balanced'` adjusts split impurity evaluations to prioritize minority class isolation.

### 2. Code
```python
from sklearn.tree import DecisionTreeClassifier

# Instantiate Decision Tree Classifier with depth constraints
dt_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)

print("Decision Tree Model Instantiated:", dt_model)
```

### 3. Expected Output & How to Read It
* Output: `DecisionTreeClassifier(class_weight='balanced', max_depth=6, min_samples_leaf=5, min_samples_split=10, random_state=42)`
* `max_depth=6` caps decision paths to a maximum of 6 nested if/else checks, keeping inference ultra-fast and preventing memory bloat.

### 4. Common Pitfalls
* Leaving `max_depth=None` (default). An unconstrained decision tree fits training noise perfectly (100% train accuracy) but overfits severely, leading to degraded performance on test data.
* Expecting scale sensitivity. Decision Trees are completely invariant to monotonic feature scaling; scaling is maintained only for uniform multi-model benchmarking.

### 5. Git Checkpoint
* **Branch:** `ml/feature-decision-tree`
* **Commit Message:** `feat(model-dt): instantiate DecisionTreeClassifier with max_depth=6 and pre-pruning bounds`

---

## Step 2: Training & Prediction Execution (`.fit()` / `.predict()`)

### 1. What & Why
Training evaluates candidate thresholds across all 28 features to construct the decision tree hierarchy. Training executes in < 0.2 seconds. Inference traverses tree nodes sequentially down to leaf predictions.

### 2. Code
```python
import time

# Fit Decision Tree on preprocessed training set
start_train = time.perf_counter()
dt_model.fit(X_train_processed, y_train)
train_time_sec = time.perf_counter() - start_train

# Execute test set inference
y_pred_dt = dt_model.predict(X_test_processed)
y_probs_dt = dt_model.predict_proba(X_test_processed)[:, 1]

print(f"Decision Tree Training completed in: {train_time_sec:.4f} seconds")
print(f"First 5 predictions: {y_pred_dt[:5]}")
print(f"First 5 purchase probabilities: {y_probs_dt[:5].round(4)}")
```

### 3. Expected Output & How to Read It
* Fast training duration (< 0.2s).
* Class probabilities in `y_probs_dt` equal the training sample class ratios present in the terminating leaf node.

### 4. Common Pitfalls
* Forgetting that leaf node class probabilities in small leaves can produce coarse step probabilities (e.g., 0/5, 1/5, 2/5).

### 5. Git Checkpoint
* **Branch:** `ml/feature-decision-tree`
* **Commit Message:** `feat(model-dt): train Decision Tree model and run test predictions`

---

## Step 3: Evaluation Metrics & Feature Importance Extraction

### 1. What & Why
In addition to classification report evaluation, Decision Trees provide native Gini feature importances via `dt_model.feature_importances_`. Visualizing top features validates business intuition regarding primary drivers of conversion intent.

### 2. Code
```python
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(y_test, y_pred_dt)
prec = precision_score(y_test, y_pred_dt, pos_label=1)
rec = recall_score(y_test, y_pred_dt, pos_label=1)
f1 = f1_score(y_test, y_pred_dt, pos_label=1)

print("=== DECISION TREE EVALUATION (max_depth=6) ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}\n")
print(classification_report(y_test, y_pred_dt, target_names=['No Purchase (0)', 'Purchase (1)']))

# Extract and display top feature importances
importances = pd.Series(dt_model.feature_importances_, index=X_train_processed.columns)
print("=== TOP 5 FEATURE IMPORTANCES ===")
print(importances.sort_values(ascending=False).head(5).round(4))
```

### 3. Expected Output & How to Read It
* Accuracy: ~0.87 - 0.89.
* Minority class (Purchase 1): Precision ~0.55 - 0.62, Recall ~0.70 - 0.78, F1 ~0.63 - 0.68.
* Feature Importance output: `PageValues` typically accounts for > 60% of total Gini impurity reduction, followed by `Month_Nov` and `ExitRates`.

### 4. Common Pitfalls
* Assuming feature importances indicate directionality (positive vs negative effect). Gini importances measure magnitude of split contribution, not directional impact.

### 5. Git Checkpoint
* **Branch:** `ml/feature-decision-tree`
* **Commit Message:** `feat(model-dt): evaluate Decision Tree performance and extract Gini feature importances`

---

## Step 4: Single-Row Inference Latency Benchmark

### 1. What & Why
Decision tree single-row inference requires traversing a small set of if/else pointer comparisons bounded by `max_depth=6`. Benchmarking confirms ultra-low operational latency.

### 2. Code
```python
import numpy as np

sample_row = X_test_processed.iloc[[0]].values

# Warm-up loop
for _ in range(100):
    _ = dt_model.predict(sample_row)

# Benchmark loop
latencies_ms = []
for _ in range(1000):
    start = time.perf_counter()
    _ = dt_model.predict(sample_row)
    end = time.perf_counter()
    latencies_ms.append((end - start) * 1000.0)

avg_latency_dt = float(np.mean(latencies_ms))
std_latency_dt = float(np.std(latencies_ms))
print(f"Decision Tree Single-Row Latency: {avg_latency_dt:.4f} ms ± {std_latency_dt:.4f} ms")
```

### 3. Expected Output & How to Read It
* Average latency: `0.01 ms` to `0.04 ms`.
* Decision Trees tie with Naive Bayes for the fastest single-row prediction latency.

### 4. Common Pitfalls
* Benchmarking un-pruned trees (`max_depth=None`), where deep tree branches incur extra pointer lookups.

### 5. Git Checkpoint
* **Branch:** `ml/feature-decision-tree`
* **Commit Message:** `feat(model-dt): benchmark single-row inference latency for Decision Tree`

---

## Step 5: Model Serialization (`decision_tree.pkl`)

### 1. What & Why
Serializing `decision_tree.pkl` exports the trained tree data structure to `/model_training/artifacts/`.

### 2. Code
```python
import os
import joblib

artifacts_dir = os.path.join("model_training", "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)

model_path = os.path.join(artifacts_dir, "decision_tree.pkl")
joblib.dump(dt_model, model_path)

file_size_kb = os.path.getsize(model_path) / 1024.0
print(f"Serialized model saved to: {model_path}")
print(f"Model File Size: {file_size_kb:.2f} KB")
```

### 3. Expected Output & How to Read It
* File size: ~10 KB to 30 KB.
* A constrained tree artifact remains extremely lightweight for production serving.

### 4. Common Pitfalls
* Forgetting to set `overwrite=True` or verify target artifact directory pathing.

### 5. Git Checkpoint
* **Branch:** `ml/feature-decision-tree`
* **Commit Message:** `feat(model-dt): serialize trained decision_tree.pkl artifact to /model_training/artifacts/`
