# 04 — Model 1: Logistic Regression

This document covers training, evaluating, latency benchmarking, and serializing **Logistic Regression** on the preprocessed Online Shoppers dataset.

---

## Step 1: Model Instantiation & Hyperparameter Setup

### 1. What & Why
Logistic Regression models class probability via the sigmoid function applied to a linear combination of features. Raising `max_iter=1000` ensures optimization convergence on scaled features, while setting `C=1.0` applies default L2 regularization to prevent overfitting on collinear features. `class_weight='balanced'` adjusts loss weights inversely proportional to class frequencies to address class imbalance.

### 2. Code
```python
from sklearn.linear_model import LogisticRegression

# Instantiate Logistic Regression model
logreg_model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver='lbfgs',
    class_weight='balanced',
    random_state=42
)

print("Logistic Regression Model Instantiated:", logreg_model)
```

### 3. Expected Output & How to Read It
* Output: `LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)`
* `C=1.0` controls regularization strength (lower values = stronger regularization). `solver='lbfgs'` handles standard L2-penalized optimization smoothly.

### 4. Common Pitfalls
* Leaving `max_iter=100` at default, which triggers `ConvergenceWarning: lbfgs failed to converge`.
* Omitting `class_weight='balanced'` when optimizing for minority class recall on imbalanced datasets.

### 5. Git Checkpoint
* **Branch:** `ml/feature-logistic-regression`
* **Commit Message:** `feat(model-logreg): instantiate Logistic Regression with balanced class weights`

---

## Step 2: Training & Prediction Execution (`.fit()` / `.predict()`)

### 1. What & Why
Fitting the model computes optimal coefficients ($\beta$ values) on `X_train_processed`. Subsequent `.predict()` calls assign class labels (0 or 1), while `.predict_proba()` computes calibrated purchase probabilities for downstream API consumption.

### 2. Code
```python
import time

# Fit model on preprocessed training set
start_train = time.perf_counter()
logreg_model.fit(X_train_processed, y_train)
train_time_sec = time.perf_counter() - start_train

# Predict class labels and probabilities on test set
y_pred_logreg = logreg_model.predict(X_test_processed)
y_probs_logreg = logreg_model.predict_proba(X_test_processed)[:, 1]

print(f"Training completed in: {train_time_sec:.4f} seconds")
print(f"First 5 predictions: {y_pred_logreg[:5]}")
print(f"First 5 purchase probabilities: {y_probs_logreg[:5].round(4)}")
```

### 3. Expected Output & How to Read It
* Training completes in < 0.5 seconds.
* `y_pred_logreg` returns binary array `[0, 0, 1, 0, 0]`.
* `y_probs_logreg` outputs float probabilities (e.g. `[0.0412, 0.1285, 0.7891, 0.0210, 0.0541]`).

### 4. Common Pitfalls
* Passing un-scaled raw features (`X_train`) to `.fit()`, causing unstable gradient steps and inaccurate coefficients.

### 5. Git Checkpoint
* **Branch:** `ml/feature-logistic-regression`
* **Commit Message:** `feat(model-logreg): train Logistic Regression model and execute test inference`

---

## Step 3: Evaluation Metrics & Classification Report Analysis

### 1. What & Why
Generating `classification_report` and standard metrics (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`) evaluates performance. Reading class 1 precision/recall specifically indicates how accurately purchase intentions are detected.

### 2. Code
```python
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Compute metrics
acc = accuracy_score(y_test, y_pred_logreg)
prec = precision_score(y_test, y_pred_logreg, pos_label=1)
rec = recall_score(y_test, y_pred_logreg, pos_label=1)
f1 = f1_score(y_test, y_pred_logreg, pos_label=1)

print("=== LOGISTIC REGRESSION EVALUATION ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}\n")
print(classification_report(y_test, y_pred_logreg, target_names=['No Purchase (0)', 'Purchase (1)']))
```

### 3. Expected Output & How to Read It
* Accuracy: ~0.84 - 0.87.
* Minority class (Purchase 1): Precision ~0.50 - 0.60, Recall ~0.70 - 0.80, F1 ~0.60.
* High recall on Class 1 demonstrates that `class_weight='balanced'` effectively captures purchasing intent sessions despite imbalance.

### 4. Common Pitfalls
* Reading overall macro accuracy alone and assuming 85% accuracy implies excellent minority class prediction.

### 5. Git Checkpoint
* **Branch:** `ml/feature-logistic-regression`
* **Commit Message:** `feat(model-logreg): evaluate Logistic Regression metrics and class performance`

---

## Step 4: Single-Row Inference Latency Benchmark

### 1. What & Why
The PRD requires surfacing per-inference latency in milliseconds for every served model. We benchmark single-row prediction latency by executing 1,000 iterations after a 100-iteration warm-up loop.

### 2. Code
```python
import numpy as np

# Select single sample row (2D array shape: 1 x n_features)
sample_row = X_test_processed.iloc[[0]].values

# Warm-up loop (100 iterations)
for _ in range(100):
    _ = logreg_model.predict(sample_row)

# Benchmark loop (1000 iterations)
latencies_ms = []
for _ in range(1000):
    start = time.perf_counter()
    _ = logreg_model.predict(sample_row)
    end = time.perf_counter()
    latencies_ms.append((end - start) * 1000.0)

avg_latency_logreg = float(np.mean(latencies_ms))
std_latency_logreg = float(np.std(latencies_ms))
print(f"Logistic Regression Single-Row Latency: {avg_latency_logreg:.4f} ms ± {std_latency_logreg:.4f} ms")
```

### 3. Expected Output & How to Read It
* Average latency: `0.02 ms` to `0.08 ms`.
* Interpretation: Extremely fast inference, serving as the low-latency baseline among all 5 models.

### 4. Common Pitfalls
* Passing a 1D vector `X_test_processed.iloc[0].values` instead of 2D matrix `iloc[[0]].values`, triggering a shape mismatch ValueError in sklearn.

### 5. Git Checkpoint
* **Branch:** `ml/feature-logistic-regression`
* **Commit Message:** `feat(model-logreg): benchmark single-row prediction latency for API contract`

---

## Step 5: Model Serialization (`logistic_regression.pkl`)

### 1. What & Why
Serializing the trained model instance to `/model_training/artifacts/logistic_regression.pkl` via `joblib` completes the training stage and prepares the artifact for FastAPI serving.

### 2. Code
```python
import os
import joblib

artifacts_dir = os.path.join("model_training", "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)

model_path = os.path.join(artifacts_dir, "logistic_regression.pkl")
joblib.dump(logreg_model, model_path)

file_size_kb = os.path.getsize(model_path) / 1024.0
print(f"Serialized model saved to: {model_path}")
print(f"Model File Size: {file_size_kb:.2f} KB")
```

### 3. Expected Output & How to Read It
* Output: `logistic_regression.pkl` created; file size ~3 KB to 10 KB.
* Compact size reflects that linear models store only lightweight coefficient vectors.

### 4. Common Pitfalls
* Forgetting to verify that `artifacts_dir` exists before calling `joblib.dump()`.

### 5. Git Checkpoint
* **Branch:** `ml/feature-logistic-regression`
* **Commit Message:** `feat(model-logreg): serialize trained logistic_regression.pkl artifact`
