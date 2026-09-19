

### 5. Git Checkpoint
* **Branch:** `ml/feature-preprocessing`
* **Commit Message:** `feat(preprocess): apply StandardScaler to numeric features on X_train only`

---

## Step 4: Transformer Serialization (`scaler.pkl` & `encoder.pkl`)

### 1. What & Why
To serve model predictions via FastAPI in production without refitting data or encountering data leakage, all fitted transformers (`scaler` and `encoder`) must be serialized directly to disk immediately after fitting.

### 2. Code
```python
import os
import joblib

ARTIFACTS_DIR = os.path.join("model_training", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
encoder_path = os.path.join(ARTIFACTS_DIR, "encoder.pkl")

# Serialize fitted transformers
joblib.dump(scaler, scaler_path)
joblib.dump(encoder, encoder_path)

print(f"Successfully saved scaler to: {scaler_path}")
print(f"Successfully saved encoder to: {encoder_path}")

# Verification check: test loading
loaded_scaler = joblib.load(scaler_path)
loaded_encoder = joblib.load(encoder_path)
print("Transformers serialization test verified successfully.")
```

### 3. Expected Output & How to Read It
* Files `scaler.pkl` and `encoder.pkl` created under `/model_training/artifacts/`.
* Re-loading files via `joblib.load()` executes without errors.

### 4. Common Pitfalls
* Storing fitted transformers only in memory during notebook execution and forgetting to save `.pkl` files for backend serving.
* Saving raw dataset transformation functions instead of serialized fitted scikit-learn estimator instances.

### 5. Git Checkpoint
* **Branch:** `ml/feature-preprocessing`
* **Commit Message:** `feat(preprocess): serialize fitted scaler and encoder to /model_training/artifacts/`
