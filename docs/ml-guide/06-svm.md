# 06 — Model 3: Support Vector Machine (SVM)

This document covers training, evaluating, latency benchmarking, and serializing **Support Vector Classification (SVC)** on the preprocessed Online Shoppers dataset.

---

## Step 1: Model Instantiation & Hyperparameter Setup

### 1. What & Why
Support Vector Machines find an optimal maximum-margin hyperplane separating classes in feature space. Using `kernel='rbf'` (Radial Basis Function) maps non-linear feature relationships (such as zero-inflated `PageValues` interactions) into higher-dimensional space. Setting `probability=True` is **mandatory** for our API contract because SVC requires internal 5-fold Platt scaling cross-validation to output class probabilities via `.predict_proba()`. Setting `class_weight='balanced'` adjusts boundary margins to account for class imbalance.

### 2. Code
```python
from sklearn.svm import SVC

# Instantiate Support Vector Classifier
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    class_weight='balanced',
    random_state=42
)

print("SVM Model Instantiated:", svm_model)
```

### 3. Expected Output & How to Read It
* Output: `SVC(C=1.0, class_weight='balanced', probability=True, random_state=42)`
* `gamma='scale'` sets $\gamma = \frac{1}{n_{features} \cdot \text{Var}(X)}$, preventing over-fitting in high-dimensional space. `probability=True` enables Platt scaling probability calibration.

### 4. Common Pitfalls
* Omitting `probability=True`. If omitted, calling `svm_model.predict_proba()` at prediction time will throw `AttributeError: predict_proba is not available when probability=False`.
* Training SVC on un-scaled features. RBF kernels rely heavily on Euclidean distances $\|x - x'\|^2$; un-scaled features cause optimization failure.

### 5. Git Checkpoint
* **Branch:** `ml/feature-svm`
* **Commit Message:** `feat(model-svm): instantiate SVC with RBF kernel, Platt scaling probability, and class balancing`

---

## Step 2: Training & Prediction Execution (`.fit()` / `.predict()`)

### 1. What & Why
Training SVC involves solving a quadratic programming problem to identify support vectors along decision margins. Because `probability=True` runs 5-fold cross-validation internally, fitting takes longer than linear models (15 to 45 seconds).

### 2. Code
```python
import time

# Fit SVC on preprocessed training set
start_train = time.perf_counter()
svm_model.fit(X_train_processed, y_train)
train_time_sec = time.perf_counter() - start_train

# Execute inference
y_pred_svm = svm_model.predict(X_test_processed)
y_probs_svm = svm_model.predict_proba(X_test_processed)[:, 1]

print(f"SVM Training completed in: {train_time_sec:.2f} seconds")
print(f"First 5 predictions: {y_pred_svm[:5]}")
print(f"First 5 purchase probabilities: {y_probs_svm[:5].round(4)}")
```

### 3. Expected Output & How to Read It
* Training time: ~10 to 45 seconds (longest training duration among all 5 models).
* Output: Calibrated continuous probabilities (e.g., `[0.0812, 0.1450, 0.8912, 0.0310, 0.0721]`).

### 4. Common Pitfalls
* Assuming SVC training is stuck if it takes ~30 seconds. Quadratic scaling $O(N^2)$ combined with Platt scaling cross-validation requires substantially more computation time.

### 5. Git Checkpoint
* **Branch:** `ml/feature-svm`
* **Commit Message:** `feat(model-svm): train SVC model with Platt scaling cross-validation`

---

## Step 3: Evaluation Metrics & Classification Report Analysis

### 1. What & Why
SVM with RBF kernel excels at learning non-linear decision surfaces. Evaluating test set metrics demonstrates how effectively the maximum-margin hyperplane handles minority class prediction.

### 2. Code
```python
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(y_test, y_pred_svm)
prec = precision_score(y_test, y_pred_svm, pos_label=1)
rec = recall_score(y_test, y_pred_svm, pos_label=1)
f1 = f1_score(y_test, y_pred_svm, pos_label=1)

print("=== SVM EVALUATION (RBF Kernel) ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}\n")
print(classification_report(y_test, y_pred_svm, target_names=['No Purchase (0)', 'Purchase (1)']))
```

### 3. Expected Output & How to Read It
* Accuracy: ~0.87 - 0.90 (frequently the highest raw accuracy among all 5 models).
* Minority Class (Purchase 1): Precision ~0.58 - 0.65, Recall ~0.72 - 0.78, F1 ~0.65 - 0.70.
* High F1 score confirms that RBF non-linear margins effectively separate purchasing intent patterns.

### 4. Common Pitfalls
* Comparing SVM metrics without checking inference speed; SVM delivers high accuracy but incurs higher compute cost.

### 5. Git Checkpoint
* **Branch:** `ml/feature-svm`
* **Commit Message:** `feat(model-svm): evaluate SVC classification performance on test set`

---

## Step 4: Single-Row Inference Latency Benchmark

### 1. What & Why
Benchmarking single-row prediction latency for SVC measures the time required to evaluate a query instance against all stored support vectors.

### 2. Code
```python
import numpy as np

sample_row = X_test_processed.iloc[[0]].values

# Warm-up loop
for _ in range(100):
    _ = svm_model.predict(sample_row)

# Benchmark loop
latencies_ms = []
for _ in range(1000):
    start = time.perf_counter()
    _ = svm_model.predict(sample_row)
    end = time.perf_counter()
    latencies_ms.append((end - start) * 1000.0)

avg_latency_svm = float(np.mean(latencies_ms))
std_latency_svm = float(np.std(latencies_ms))
print(f"SVM Single-Row Latency: {avg_latency_svm:.4f} ms ± {std_latency_svm:.4f} ms")
```

### 3. Expected Output & How to Read It
* Average latency: `0.8 ms` to `2.5 ms`.
* Interpretation: Moderate latency — faster than KNN (~10 ms), but 20x to 50x slower than Logistic Regression (~0.04 ms).

### 4. Common Pitfalls
* Measuring `predict_proba()` timing instead of `predict()` when comparing raw model latency across algorithms.

### 5. Git Checkpoint
* **Branch:** `ml/feature-svm`
* **Commit Message:** `feat(model-svm): benchmark single-row inference latency for SVC`

---

## Step 5: Model Serialization (`svm.pkl`)

### 1. What & Why
Serializing `svm.pkl` exports the support vectors and dual coefficients required for serving predictions.

### 2. Code
```python
import os
import joblib

artifacts_dir = os.path.join("model_training", "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)

model_path = os.path.join(artifacts_dir, "svm.pkl")
joblib.dump(svm_model, model_path)

file_size_kb = os.path.getsize(model_path) / 1024.0
print(f"Serialized model saved to: {model_path}")
print(f"Model File Size: {file_size_kb:.2f} KB ({file_size_kb/1024.0:.2f} MB)")
```

### 3. Expected Output & How to Read It
* File size: ~500 KB to 2,000 KB (0.5 MB - 2 MB).
* Size explanation: SVC retains a subset of training data rows designated as support vectors (~2,000 to 4,000 vectors).

### 4. Common Pitfalls
* Not verifying write permissions in `artifacts_dir`.

### 5. Git Checkpoint
* **Branch:** `ml/feature-svm`
* **Commit Message:** `feat(model-svm): serialize trained svm.pkl artifact to /model_training/artifacts/`
