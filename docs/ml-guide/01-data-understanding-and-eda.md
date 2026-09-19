# 01 — Data Understanding & Exploratory Data Analysis (EDA)

This document guides you through loading and exploring the **UCI Online Shoppers Purchasing Intention Dataset (ID 468)**. The dataset contains 12,330 user sessions, 17 predictor features, and 1 binary target (`Revenue`).

---

## Step 1: Dataset Ingestion (API & Local CSV Fallback)

### 1. What & Why
Loading the dataset programmatically via `ucimlrepo` provides direct access to canonical metadata and features while establishing a reproducible fallback to local raw CSV storage (`model_training/data/raw/online_shoppers_intention.csv`). Securing both loading mechanisms ensures offline development stability and automated data fetching correctness.

### 2. Code
```python
import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Define local storage target
def load_shoppers_dataset() -> pd.DataFrame:
    """Fetch from UCI ML repo or load from local CSV fallback."""
    if os.path.exists(CSV_PATH):
        print(f"Loading dataset from local CSV: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
    else:
        print("Fetching dataset from UCI ML Repository (ID 468)...")
        dataset = fetch_ucirepo(id=468)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
        # Save local copy for offline stability
        df.to_csv(CSV_PATH, index=False)
        print(f"Saved local copy to {CSV_PATH}")
    return df

df = load_shoppers_dataset()
print("Dataset Shape:", df.shape)
```

### 3. Expected Output & How to Read It
* Output: `Dataset Shape: (12330, 18)`
* `12330` total rows (sessions) and `18` columns (17 features + 1 target column `Revenue`).

### 4. Common Pitfalls
* Assuming `ucimlrepo` will always have internet connectivity. Failing to write the fallback CSV step will break offline script execution.
* Forgetting that `ucimlrepo` returns `X` and `y` separately as dataframes; concatenating them improperly can mismatch index rows if indices are reordered.

### 5. Git Checkpoint
* **Branch:** `ml/feature-eda`
* **Commit Message:** `feat(eda): add dataset ingestion module with local CSV fallback`

---

## Step 2: Structural Audit & Missing Value Verification

### 1. What & Why
Conducting a baseline structural inspection (`.info()`, `.describe()`, `.isnull().sum()`) confirms data types (numerical vs categorical), missing value counts, and range anomalies. Verifying zero missing values early avoids unnecessary imputation steps during preprocessing.

### 2. Code
```python
# Display dataset summary info and null counts
print("=== DATASET INFO ===")
df.info()

print("\n=== MISSING VALUES PER COLUMN ===")
null_counts = df.isnull().sum()
print(null_counts[null_counts > 0] if null_counts.sum() > 0 else "Zero missing values detected across all columns.")

print("\n=== NUMERICAL FEATURES SUMMARY ===")
print(df.describe().T[['mean', 'std', 'min', '50%', 'max']])
```

### 3. Expected Output & How to Read It
* Data Types: 10 numerical features (`Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`), 7 categorical/boolean (`Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`), and 1 target (`Revenue`).
* `df.isnull().sum()` outputs zero missing values across all 12,330 entries.
* `.describe()` shows high variance in durations (e.g. `ProductRelated_Duration` ranges from `0.0` to `63973.5` seconds), while rates (`BounceRates`, `ExitRates`) are bounded between `0.0` and `0.2`.

### 4. Common Pitfalls
* Misinterpreting integer-coded categoricals (such as `OperatingSystems`, `Browser`, `Region`, `TrafficType`) as continuous numerical variables due to their integer dtypes in pandas.
* Overlooking zero-inflated distributions (`PageValues` has a 50th percentile of `0.0` because most non-purchasing sessions view pages with 0 analytics valuation).

### 5. Git Checkpoint
* **Branch:** `ml/feature-eda`
* **Commit Message:** `feat(eda): complete structural audit and missing value check`

---

## Step 3: Class Imbalance Analysis (`Revenue`)

### 1. What & Why
Checking the target distribution (`Revenue`) is critical because accuracy alone is misleading on imbalanced data. A trivial model predicting `False` for every single session would achieve ~84.5% raw accuracy while being completely useless for business conversion detection.

### 2. Code
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Compute target class breakdown
counts = df['Revenue'].value_counts()
proportions = df['Revenue'].value_counts(normalize=True) * 100

print(f"False (No Purchase): {counts[False]} ({proportions[False]:.2f}%)")
print(f"True (Purchase):    {counts[True]} ({proportions[True]:.2f}%)")

plt.figure(figsize=(6, 4))
ax = sns.countplot(x='Revenue', data=df, palette=['#e74c3c', '#2ecc71'])
plt.title("Class Imbalance: Target Column (Revenue)")
plt.xlabel("Session Ended in Purchase (Revenue)")
plt.ylabel("Number of Sessions")
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)', 
                (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold')
plt.tight_layout()
plt.show()
```

### 3. Expected Output & How to Read It
* Output: `False: 10,422 (84.53%)`, `True: 1,908 (15.47%)`.
* The bar chart visually highlights a ~5.5:1 ratio favoring non-purchasing sessions.
* Interpretation: Precision, Recall, and F1-score evaluated specifically on the minority class (`Revenue = True`) must be our primary optimization metrics, rather than overall raw accuracy. Stratified splitting is mandatory.

### 4. Common Pitfalls
* Relying on `accuracy_score` to evaluate models trained on this dataset.
* Performing random train-test splits without `stratify=y`, which risks severe class proportion skew between train and test sets.

### 5. Git Checkpoint
* **Branch:** `ml/feature-eda`
* **Commit Message:** `feat(eda): add target class imbalance analysis and visualization`

---

## Step 4: Univariate Feature Breakdown

### 1. What & Why
Univariate exploration helps identify feature skewness, scale disparities, and categorical cardinality. Understanding feature scales informs why algorithms sensitive to distance metrics (KNN, SVM) require normalization, whereas tree-based models (Decision Trees) do not.

### 2. Code
```python
# List feature groupings explicitly
num_cols = [
    'Administrative', 'Administrative_Duration', 'Informational', 
    'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
]
cat_cols = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']

# Histograms of key continuous numerical features
df[num_cols].hist(bins=30, figsize=(15, 10), color='#3498db', edgecolor='black')
plt.suptitle("Univariate Distribution of Numerical Features", fontsize=14)
plt.tight_layout()
plt.show()

# Unique counts for categoricals
for col in cat_cols:
    print(f"Categorical '{col}' unique values count: {df[col].nunique()} -> Values: {df[col].unique()}")
```

### 3. Expected Output & How to Read It
* All duration features (`Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration`) exhibit heavy right-skew with long tails.
* `PageValues` exhibits an extreme zero spike (~78% of rows are `0.0`).
* Categoricals: `Month` has 10 unique string values (e.g. 'Feb', 'Mar', 'May', 'Oct'); `VisitorType` has 3 ('Returning_Visitor', 'New_Visitor', 'Other'); `Weekend` is boolean.

### 4. Common Pitfalls
* Treating right-skewed counts as Gaussian distributions without checking scale magnitude.
* Not identifying string categorical columns (`Month`, `VisitorType`) that will break numerical matrix operations if left unencoded.

### 5. Git Checkpoint
* **Branch:** `ml/feature-eda`
* **Commit Message:** `feat(eda): add univariate distribution plots and categorical cardinality audit`

---

## Step 5: Bivariate Feature Exploration vs Target

### 1. What & Why
Bivariate analysis explores how individual predictor features correlate with the purchase outcome (`Revenue`). Identifying high-signal features (such as `PageValues`) helps validate domain intuition before feeding data into models.

### 2. Code
```python
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x='Revenue', y='PageValues', data=df, palette=['#e74c3c', '#2ecc71'])
plt.title("PageValues by Purchase Intent (Revenue)")

plt.subplot(1, 2, 2)
sns.barplot(x='Month', y='Revenue', data=df, order=['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], ci=None, palette='viridis')
plt.title("Conversion Rate (Revenue = True) by Month")
plt.ylabel("Conversion Rate")

plt.tight_layout()
plt.show()
```

### 3. Expected Output & How to Read It
* `PageValues` boxplot: Sessions ending in `Revenue = True` show significantly higher `PageValues` medians and 75th percentiles than non-converting sessions.
* `Month` conversion bar chart: November (`Nov`) has the highest volume of purchasing sessions, while February (`Feb`) has extremely low conversion rates.

### 4. Common Pitfalls
* Overlooking temporal seasonality effects captured in `Month`.
* Ignoring outliers in boxplots that reflect genuine power-user behavior rather than measurement errors.

### 5. Git Checkpoint
* **Branch:** `ml/feature-eda`
* **Commit Message:** `feat(eda): add bivariate analysis of PageValues and Month vs target`

---

## Step 6: Feature Correlation Heatmap & Naive Bayes Risk Flagging

### 1. What & Why
Computing a correlation matrix across numerical features exposes multicollinearity. High correlation between features (e.g. `BounceRates` and `ExitRates` with correlation > 0.90) violates Naive Bayes' fundamental assumption of conditional independence, providing a clear pedagogical point for algorithm comparison.

### 2. Code
```python
plt.figure(figsize=(10, 8))
corr_matrix = df[num_cols].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title("Correlation Heatmap: Numerical Features")
plt.tight_layout()
plt.show()

# Flag correlation pairs exceeding threshold > 0.70
print("=== HIGH MULTICOLLINEARITY PAIRS (> 0.70) ===")
for i in range(len(num_cols)):
    for j in range(i+1, len(num_cols)):
        if abs(corr_matrix.iloc[i, j]) > 0.70:
            print(f"High correlation between '{num_cols[i]}' and '{num_cols[j]}': {corr_matrix.iloc[i, j]:.3f}")
```

### 3. Expected Output & How to Read It
* Output flags: `BounceRates` and `ExitRates` show strong positive correlation (~0.91). `ProductRelated` and `ProductRelated_Duration` show correlation ~0.86.
* Interpretation: `BounceRates` and `ExitRates` contain redundant information. Distance models (KNN) and Naive Bayes will be affected by feature redundancy unless properly handled or scaled.

### 4. Common Pitfalls
* Automatically dropping highly correlated features before testing tree-based models, which natively handle redundant inputs without performance loss.
* Including categorical un-encoded columns in `df.corr()`, which triggers string conversion errors or silent omission.

### 5. Git Checkpoint
* **Branch:** `ml/feature-eda`
* **Commit Message:** `feat(eda): generate feature correlation heatmap and flag multicollinearity for preprocessing`
