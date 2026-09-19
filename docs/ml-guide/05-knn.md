# 05 — Model 2: K-Nearest Neighbors (KNN)

This document covers training, evaluating, latency benchmarking, and serializing **K-Nearest Neighbors (KNN)** on the preprocessed Online Shoppers dataset.

---

## Step 1: Model Instantiation & Hyperparameter Setup

### 1. What & Why
KNN is a non-parametric, distance-based algorithm that classifies new data points based on majority vote among its $k$ closest neighbors in Euclidean feature space. Setting `n_neighbors=5` strikes a balance between noise sensitivity and over-smoothing. On an imbalanced dataset (84.5% False), setting `k` too high (e.g. $k=50$) causes neighbor neighborhoods to be dominated by majority non-purchasers, driving class 1 recall to zero. Setting `n_jobs=-1` enables parallel distance computations across CPU cores.

### 2. Code
```python
from sklearn.neighbors import KNeighborsClassifier

# Instantiate KNN Classifier
knn_model = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    metric='minkowski',  # Euclidean distance when p=2
    p=2,
    n_jobs=-1
)

print("KNN Model Instantiated:", knn_model)
```

### 3. Expected Output & How to Read It
* Output: `KNeighborsClassifier(n_jobs=-1, n_neighbors=5)`
* `weights='uniform'` treats all $k$ nearest neighbors equally regardless of distance; `metric='minkowski'` with `p=2` executes standard Euclidean distance comparisons.

### 4. Common Pitfalls
* Setting `n_neighbors` to a large number (e.g. 50+) on an imbalanced dataset, which leads to total suppression of minority class predictions.
* Training KNN without scaling numerical features. Un-scaled continuous durations (0–60,000) completely drown out bounded rate metrics (0–0.2), causing distance calculations to depend solely on duration columns.

### 5. Git Checkpoint
* **Branch:** `ml/feature-knn`
* **Commit Message:** `feat(model-knn): instantiate KNeighborsClassifier with k=5 and parallel jobs`

---

## Step 2: Training & Prediction Execution (`.fit()` / `.predict()`)

### 1. What & Why
KNN is a "lazy learner." Calling `.fit()` simply indexes the training data matrix in memory without constructing an explicit parametric model. Inference (`.predict()`) requires calculating distance from the target instance to every stored training sample.

### 2. Code
```python
import time

# Fit model (stores training dataset in spatial data structure)
start_train = time.perf_counter()
knn_model.fit(X_train_processed, y_train)
train_time_sec = time.perf_counter() - start_train

# Predict class labels and class probabilities
y_pred_knn = knn_model.predict(X_test_processed)
y_probs_knn = knn_model.predict_proba(X_test_processed)[:, 1]

print(f"KNN 'Training' (indexing) completed in: {train_time_sec:.4f} seconds")
print(f"First 5 predictions: {y_pred_knn[:5]}")
print(f"First 5 purchase probabilities: {y_probs_knn[:5].round(4)}")
```

### 3. Expected Output & How to Read It
* Training time is nearly instantaneous (< 0.1 seconds).
* `y_probs_knn` values take discrete step proportions (e.g., `0.0`, `0.2`, `0.4`, `0.6`, `0.8`, `1.0`) because probabilities equal the fraction of 5 neighbors belonging to Class 1.

### 4. Common Pitfalls
* Assuming fast training time implies fast inference time. KNN defers all computational work to prediction time.

### 5. Git Checkpoint
* **Branch:** `ml/feature-knn`
* **Commit Message:** `feat(model-knn): execute KNN fitting and test dataset predictions`

---

## Step 3: Evaluation Metrics & Classification Report Analysis

### 1. What & Why
Evaluating KNN performance across accuracy, precision, recall, and F1 reveals how well spatial proximity in standardized feature space maps to purchasing intent.

### 2. Code
```python
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(y_test, y_pred_knn)
prec = precision_score(y_test, y_pred_knn, pos_label=1)
rec = recall_score(y_test, y_pred_knn, pos_label=1)
f1 = f1_score(y_test, y_pred_knn, pos_label=1)

print("=== KNN EVALUATION (k=5) ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}\n")
print(classification_report(y_test, y_pred_knn, target_names=['No Purchase (0)', 'Purchase (1)']))
```

### 3. Expected Output & How to Read It
* Accuracy: ~0.86 - 0.88.
* Minority class (Purchase 1): Precision ~0.60 - 0.68, Recall ~0.35 - 0.45, F1 ~0.48 - 0.55.
* Interpretation: KNN achieves respectable precision but lower recall on class 1 compared to balanced Logistic Regression because unweighted majority voting favors majority class neighbors.

### 4. Common Pitfalls
* Expecting high recall without applying distance weighting (`weights='distance'`) or threshold tuning.

### 5. Git Checkpoint
* **Branch:** `ml/feature-knn`
* **Commit Message:** `feat(model-knn): evaluate KNN classification metrics on test set`

---

## Step 4: Single-Row Inference Latency Benchmark

### 1. What & Why
Measuring single-row inference latency for KNN highlights its primary operational trade-off: because KNN must calculate Euclidean distance against 9,864 training rows for every prediction, single-row latency is significantly higher than linear or tree-based models.

### 2. Code
```python
import numpy as np

sample_row = X_test_processed.iloc[[0]].values

# Warm-up loop
for _ in range(100):
    _ = knn_model.predict(sample_row)

# Benchmark loop
latencies_ms = []
for _ in range(1000):
    start = time.perf_counter()
    _ = knn_model.predict(sample_row)
    end = time.perf_counter()
    latencies_ms.append((end - start) * 1000.0)

avg_latency_knn = float(np.mean(latencies_ms))
std_latency_knn = float(np.std(latencies_ms))
print(f"KNN Single-Row Latency: {avg_latency_knn:.4f} ms ± {std_latency_knn:.4f} ms")
```

### 3. Expected Output & How to Read It
* Average latency: `5.0 ms` to `15.0 ms` (compared to `0.03 ms` for Logistic Regression).
* Interpretation: KNN is ~100x to 300x slower per request at inference time. This is a core trade-off documented in Algorithm Arena.

### 4. Common Pitfalls
* Running benchmark without `n_jobs=-1`, which inflates latency on multi-core systems.

### 5. Git Checkpoint
* **Branch:** `ml/feature-knn`
* **Commit Message:** `feat(model-knn): benchmark single-row inference latency showcasing KNN latency trade-off`

---

## Step 5: Model Serialization (`knn.pkl`)

### 1. What & Why
Serializing `knn.pkl` saves the entire training dataset matrix inside the pickle file. This produces a noticeably larger file artifact than parametric models, illustrating storage cost trade-offs.

### 2. Code
```python
import os
import joblib

artifacts_dir = os.path.join("model_training", "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)

model_path = os.path.join(artifacts_dir, "knn.pkl")
joblib.dump(knn_model, model_path)

file_size_kb = os.path.getsize(model_path) / 1024.0
print(f"Serialized model saved to: {model_path}")
print(f"Model File Size: {file_size_kb:.2f} KB ({file_size_kb/1024.0:.2f} MB)")
```

### 3. Expected Output & How to Read It
* File size: ~1,500 KB to 3,000 KB (1.5 MB - 3 MB).
* Contrast: Logistic Regression is ~4 KB. KNN is ~500x larger because it retains all 9,864 training feature vectors.

### 4. Common Pitfalls
* Surprised by large file size and assuming a bug occurred. KNN serialization size is naturally proportional to dataset size $N \times D$.

### 5. Git Checkpoint
* **Branch:** `ml/feature-knn`
* **Commit Message:** `feat(model-knn): serialize trained knn.pkl artifact to /model_training/artifacts/`
