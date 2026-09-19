# 08 — Model 5: Naive Bayes

This document covers training, evaluating, latency benchmarking, and serializing a **Gaussian Naive Bayes** model on the preprocessed Online Shoppers dataset.

---

## Step 1: Model Instantiation & Hyperparameter Setup

### 1. What & Why
Gaussian Naive Bayes applies Bayes' Theorem with the "naive" assumption of conditional independence between every pair of features given the class label: $P(X_1, X_2 | Y) = P(X_1 | Y) \cdot P(X_2 | Y)$. `GaussianNB` models continuous numeric features using Gaussian normal distributions. The primary hyperparameter `var_smoothing=1e-9` adds a small fraction of the largest feature variance to feature variances during probability estimation, preventing division-by-zero errors when feature variance approaches zero.

### 2. Code
```python
from sklearn.naive_bayes import GaussianNB

# Instantiate Gaussian Naive Bayes Classifier
nb_model = GaussianNB(
    var_smoothing=1e-9
)

print("Naive Bayes Model Instantiated:", nb_model)
```

### 3. Expected Output & How to Read It
* Output: `GaussianNB(var_smoothing=1e-09)`
* GaussianNB has virtually no hyperparameter knobs to tune. It computes class priors $P(Y)$ and per-feature means $\mu_{i,k}$ and variances $\sigma^2_{i,k}$ directly from training data.

### 4. Common Pitfalls
* Using `CategoricalNB` or `MultinomialNB` directly on scaled continuous values that contain negative numbers (StandardScaler produces negative z-scores). `GaussianNB` is required when processing continuous normalized inputs.

### 5. Git Checkpoint
* **Branch:** `ml/feature-naive-bayes`
* **Commit Message:** `feat(model-nb): instantiate GaussianNB with default variance smoothing`

---

## Step 2: Training & Prediction Execution (`.fit()` / `.predict()`)

### 1. What & Why
Fitting `GaussianNB` calculates summary statistics (mean and variance per feature per class) in a single pass over `X_train_processed`. Training is virtually instantaneous (< 0.05s). Inference evaluates Gaussian probability density functions to calculate posterior class probabilities.

### 2. Code
```python
import time

# Fit GaussianNB on preprocessed training set
start_train = time.perf_counter()
nb_model.fit(X_train_processed, y_train)
train_time_sec = time.perf_counter() - start_train

# Execute test set inference
y_pred_nb = nb_model.predict(X_test_processed)
y_probs_nb = nb_model.predict_proba(X_test_processed)[:, 1]

print(f"Naive Bayes Training completed in: {train_time_sec:.4f} seconds")
print(f"First 5 predictions: {y_pred_nb[:5]}")
print(f"First 5 purchase probabilities: {y_probs_nb[:5].round(4)}")
```

### 3. Expected Output & How to Read It
* Fast execution (< 0.05 seconds).
* Output: Posterior probabilities $P(Revenue=True | X)$. Naive Bayes probabilities frequently push toward extreme 0.0 or 1.0 values due to multiplying many conditional probabilities together under the independence assumption.

### 4. Common Pitfalls
* Interpreting uncalibrated Naive Bayes output probabilities as true real-world percentages. Because feature independence is violated, raw NB probabilities tend to be overconfident (clustered near 0 or 1).

### 5. Git Checkpoint
* **Branch:** `ml/feature-naive-bayes`
* **Commit Message:** `feat(model-nb): train GaussianNB model and compute posterior test probabilities`

---

## Step 3: Evaluation Metrics & Conditional Independence Impact Analysis

### 1. What & Why
Evaluating Naive Bayes illustrates how domain feature properties affect model performance. In EDA (Document 01), we discovered that `BounceRates` and `ExitRates` exhibit high correlation ($r > 0.90$), as do `ProductRelated` and `ProductRelated_Duration` ($r > 0.86$). Because Naive Bayes assumes these features are independent given the class, it double-counts evidence from correlated pairs, leading to reduced precision compared to decision trees or logistic regression.

### 2. Code
```python
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(y_test, y_pred_nb)
prec = precision_score(y_test, y_pred_nb, pos_label=1)
rec = recall_score(y_test, y_pred_nb, pos_label=1)
f1 = f1_score(y_test, y_pred_nb, pos_label=1)

print("=== NAIVE BAYES EVALUATION (GaussianNB) ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}\n")
print(classification_report(y_test, y_pred_nb, target_names=['No Purchase (0)', 'Purchase (1)']))
```

### 3. Expected Output & How to Read It
* Accuracy: ~0.78 - 0.82 (typically lower than tree and SVM models).
* Minority Class (Purchase 1): Precision ~0.35 - 0.45, Recall ~0.65 - 0.75, F1 ~0.48 - 0.55.
* Interpretation: High recall paired with lower precision reflects that double-counting correlated activity signals causes Naive Bayes to flag moderate-activity sessions as purchases, generating higher false positive rates.

### 4. Common Pitfalls
* Assuming Naive Bayes is defective. Lower precision directly demonstrates the theoretical constraint of Naive Bayes when applied to correlated real-world features.

### 5. Git Checkpoint
* **Branch:** `ml/feature-naive-bayes`
* **Commit Message:** `feat(model-nb): evaluate GaussianNB metrics and analyze independence assumption violations`

---

## Step 4: Single-Row Inference Latency Benchmark

### 1. What & Why
Measuring single-row prediction latency confirms that closed-form Gaussian PDF evaluation yields near-instantaneous inference speed.

### 2. Code
```python
import numpy as np

sample_row = X_test_processed.iloc[[0]].values

# Warm-up loop
for _ in range(100):
    _ = nb_model.predict(sample_row)

# Benchmark loop
latencies_ms = []
for _ in range(1000):
    start = time.perf_counter()
    _ = nb_model.predict(sample_row)
    end = time.perf_counter()
    latencies_ms.append((end - start) * 1000.0)

avg_latency_nb = float(np.mean(latencies_ms))
std_latency_nb = float(np.std(latencies_ms))
print(f"Naive Bayes Single-Row Latency: {avg_latency_nb:.4f} ms ± {std_latency_nb:.4f} ms")
```

### 3. Expected Output & How to Read It
* Average latency: `0.01 ms` to `0.03 ms`.
* Naive Bayes provides ultra-fast single-row prediction speeds.

### 4. Common Pitfalls
* Timing string preprocessing overhead inside the latency loop instead of passing the pre-transformed numpy row directly.

### 5. Git Checkpoint
* **Branch:** `ml/feature-naive-bayes`
* **Commit Message:** `feat(model-nb): benchmark single-row inference latency for GaussianNB`

---

## Step 5: Model Serialization (`naive_bayes.pkl`)

### 1. What & Why
Serializing `naive_bayes.pkl` exports the class priors, feature means, and variances to `/model_training/artifacts/`.

### 2. Code
```python
import os
import joblib

artifacts_dir = os.path.join("model_training", "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)

model_path = os.path.join(artifacts_dir, "naive_bayes.pkl")
joblib.dump(nb_model, model_path)

file_size_kb = os.path.getsize(model_path) / 1024.0
print(f"Serialized model saved to: {model_path}")
print(f"Model File Size: {file_size_kb:.2f} KB")
```

### 3. Expected Output & How to Read It
* File size: ~2 KB to 5 KB.
* Naive Bayes produces one of the smallest serialized file artifacts alongside Logistic Regression because it stores only summary Gaussian parameter vectors.

### 4. Common Pitfalls
* Forgetting to verify that `naive_bayes.pkl` is saved to `/model_training/artifacts/`.

### 5. Git Checkpoint
* **Branch:** `ml/feature-naive-bayes`
* **Commit Message:** `feat(model-nb): serialize trained naive_bayes.pkl artifact to /model_training/artifacts/`
