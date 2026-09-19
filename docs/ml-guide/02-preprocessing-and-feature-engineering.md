# 02 — Preprocessing & Feature Engineering

This document guides you through constructing the deterministic data transformation pipeline for the **UCI Online Shoppers Purchasing Intention Dataset**.

---

## Step 1: Categorical Encoding Strategy

### 1. What & Why
ML models require numerical input matrices. Categorical columns (`Month`, `VisitorType`, `Weekend`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`) must be encoded into numerical values. We use `OneHotEncoder` with `sparse_output=False` and `handle_unknown='ignore'` for non-ordinal categoricals (`Month`, `VisitorType`) and direct boolean conversion for `Weekend`. This provides a uniform feature space across all 5 models without imposing artificial numerical ordering.

### 2. Code
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def create_encoder(df: pd.DataFrame, cat_cols: list) -> OneHotEncoder:
    """Instantiate and fit OneHotEncoder on categorical columns."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(df[cat_cols])
    return encoder

cat_cols = ['Month', 'VisitorType']
df['Weekend'] = df['Weekend'].astype(int) # Boolean to 0/1 integer flag
df['Revenue'] = df['Revenue'].astype(int) # Target to 0/1 integer flag

encoder = create_encoder(df, cat_cols)
encoded_cats = encoder.transform(df[cat_cols])
encoded_feature_names = encoder.get_feature_names_out(cat_cols)
df_encoded = pd.DataFrame(encoded_cats, columns=encoded_feature_names, index=df.index)
print(f"Encoded Categorical Columns Count: {len(encoded_feature_names)}")
print("Categories sample names:", list(encoded_feature_names[:5]))
```

### 3. Expected Output & How to Read It
* Output: `Encoded Categorical Columns Count: 13` (10 months + 3 visitor types).
* `df_encoded` contains binary 0/1 columns such as `Month_Nov`, `VisitorType_Returning_Visitor`.
* `Weekend` is converted to standard 0/1 representation.

### 4. Common Pitfalls
* Using `LabelEncoder` or `OrdinalEncoder` on unordered categories like `Month` or `VisitorType`, which misleads linear models into assuming `Nov > Feb` or `New_Visitor > Returning_Visitor`.
* Forgetting `handle_unknown='ignore'`, which causes API crashes at serving time if an unseen category is submitted in a request.

### 5. Git Checkpoint
* **Branch:** `ml/feature-preprocessing`
* **Commit Message:** `feat(preprocess): implement OneHotEncoder for non-ordinal categorical features`

---

## Step 2: Stratified Train-Test Splitting

### 1. What & Why
Because `Revenue` exhibits an 84.5% / 15.5% class imbalance, standard random train-test splitting risks uneven class distribution across subsets. Using `train_test_split` with `stratify=y` guarantees that both training (80%) and testing (20%) splits maintain the exact ~15.5% positive target ratio.

### 2. Code
```python
from sklearn.model_selection import train_test_split

# Define feature subset (Numerical + Integer Categoricals + One-Hot Encoded)
num_cols = [
    'Administrative', 'Administrative_Duration', 'Informational', 
    'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
    'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Weekend'
]

X = pd.concat([df[num_cols], df_encoded], axis=1)
y = df['Revenue']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"X_train shape: {X_train.shape}, y_train conversion rate: {y_train.mean():.4f}")
print(f"X_test shape:  {X_test.shape}, y_test conversion rate:  {y_test.mean():.4f}")
```

### 3. Expected Output & How to Read It
* Output:
  `X_train shape: (9864, 28), y_train conversion rate: 0.1547`
  `X_test shape:  (2466, 28), y_test conversion rate:  0.1547`
* `X_train` has 9,864 rows; `X_test` has 2,466 rows. Both splits show an identical conversion rate of ~15.47%.

### 4. Common Pitfalls
* Fitting transformers on `X` before performing `train_test_split`. Scaling or encoding before splitting introduces **data leakage** from test set into training set.
* Omitting `random_state=42`, which makes train/test splitting non-reproducible.

### 5. Git Checkpoint
* **Branch:** `ml/feature-preprocessing`
* **Commit Message:** `feat(preprocess): execute stratified train-test split on imbalanced target`

---

## Step 3: Numeric Feature Scaling (`StandardScaler`)

### 1. What & Why
Numerical features in this dataset have radically different scales: `ProductRelated_Duration` reaches >60,000 while `BounceRates` is bounded between 0.0 and 0.2. Distance-sensitive models (KNN, SVM, Logistic Regression) fail or converge extremely slowly without feature normalization. Decision Trees and Naive Bayes are scale-invariant, but scaling all features uniformly ensures compatibility across all 5 models.

### 2. Code
```python
from sklearn.preprocessing import StandardScaler

# Columns requiring scaling (continuous & count variables)
scale_cols = [
    'Administrative', 'Administrative_Duration', 'Informational', 
    'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
]

scaler = StandardScaler()

# FIT on training set numerical columns ONLY
X_train_scaled_num = scaler.fit_transform(X_train[scale_cols])
X_test_scaled_num = scaler.transform(X_test[scale_cols])

# Replace original numerical columns with scaled values
X_train_processed = X_train.copy()
X_test_processed = X_test.copy()

X_train_processed[scale_cols] = X_train_scaled_num
X_test_processed[scale_cols] = X_test_scaled_num

print("Scaled Numerical Means (Training):", np.round(X_train_processed[scale_cols].mean().values, 2))
print("Scaled Numerical Stds (Training):", np.round(X_train_processed[scale_cols].std().values, 2))
```

### 3. Expected Output & How to Read It
* Means for scaled numerical features in training set equal `0.00`; standard deviations equal `1.00`.
* `X_train_processed` now contains standardized numerical values alongside binary encoded categorical flags.

### 4. Common Pitfalls
* Calling `scaler.fit_transform(X_test[scale_cols])` on the test set. The test set MUST only be transformed using parameters fitted on `X_train`.
* Scaling one-hot encoded binary variables (0/1), which destroys sparse indicator semantics and interpretable zero baselines.

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
