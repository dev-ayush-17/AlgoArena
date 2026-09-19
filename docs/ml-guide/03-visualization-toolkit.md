# 03 — Visualization Toolkit

This document provides a reusable visualization library for evaluating and interpreting ML models trained on the **Online Shoppers Purchasing Intention Dataset**.

---

## Tool 1: Confusion Matrix Heatmap

### 1. What & Why
A confusion matrix visualizes True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN). On an imbalanced dataset, checking raw numbers and normalized rates per class reveals whether a model is accurately identifying minority-class purchasers (`Revenue = True`) or silently failing by predicting majority `False`.

### 2. Code
```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, model_name: str):
    """Plot formatted confusion matrix with raw counts and normalized ratios."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Purchase', 'Purchase'],
                yticklabels=['No Purchase', 'Purchase'])
    
    # Overlay percentages
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.7, f"({cm_norm[i, j]*100:.1f}%)",
                    ha='center', va='center', color='red' if cm[i, j] < 500 else 'black')
            
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual True Label")
    plt.tight_layout()
    plt.show()
```

### 3. Expected Output & How to Read It
* Bottom-right cell (TP): Sessions correctly predicted as purchases.
* Bottom-left cell (FN): Purchasing sessions missed by the model.
* Top-right cell (FP): Non-purchasing sessions wrongly predicted as purchases.
* What "good" looks like: High percentage in TP (>60% normalized) with low False Negative rates on class 1.

### 4. Common Pitfalls
* Confusing row vs column labels (sklearn puts True labels on y-axis, Predicted labels on x-axis).
* Reading unnormalized counts alone without accounting for class ratio imbalance.

### 5. Git Checkpoint
* **Branch:** `ml/feature-eda`
* **Commit Message:** `feat(eda): add confusion matrix heatmap plotting utility`

---

## Tool 2: Receiver Operating Characteristic (ROC) Curve

### 1. What & Why
The ROC curve plots True Positive Rate (TPR) against False Positive Rate (FPR) across all probability thresholds. The Area Under Curve (ROC-AUC) summarizes ranking quality across threshold variations.

### 2. Code
```python
from sklearn.metrics import roc_curve, roc_auc_score

def plot_roc_curve(y_true, y_probs, model_name: str):
    """Plot ROC curve and calculate ROC-AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#2980b9', lw=2, label=f'{model_name} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title(f'ROC Curve — {model_name}')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
```

### 3. Expected Output & How to Read It
* Curve hugging upper-left corner indicates strong classification power.
* ROC-AUC score ranges from 0.5 (random guessing) to 1.0 (perfect prediction). Good score: > 0.85.

### 4. Common Pitfalls
* Over-relying on ROC-AUC for imbalanced datasets. Because FPR denominator is total negatives (10,422), large increases in False Positives produce small changes in FPR, making ROC look overly optimistic.

### 5. Git Checkpoint
* **Branch:** `ml/feature-eda`
* **Commit Message:** `feat(eda): add ROC curve plotting utility`

---

## Tool 3: Precision-Recall (PR) Curve

### 1. What & Why
The Precision-Recall curve plots Precision vs Recall for class 1 across all thresholds. Unlike ROC, PR curves do NOT include True Negatives in their calculation, making them significantly more informative and stringent for imbalanced targets.

### 2. Code
```python
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_precision_recall_curve(y_true, y_probs, model_name: str):
    """Plot Precision-Recall curve and compute Average Precision (AP)."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    baseline = (y_true == 1).sum() / len(y_true)
    
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='#e67e22', lw=2, label=f'{model_name} (AP = {ap:.3f})')
    plt.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline ({baseline:.3f})')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (Positive Predictive Value)')
    plt.title(f'Precision-Recall Curve — {model_name}')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
```

### 3. Expected Output & How to Read It
* Baseline horizontal line represents random guessing (~0.155 given our class ratio).
* What "good" looks like: Curve pushing toward top-right corner, maintaining high Precision even as Recall increases. AP > 0.65 is strong on this dataset.

### 4. Common Pitfalls
* Expecting baseline to start at (0,0). Random guess baseline equals minority class proportion ($P / N$).

### 5. Git Checkpoint
* **Branch:** `ml/feature-eda`
* **Commit Message:** `feat(eda): add Precision-Recall curve plotting utility`

---

## Tool 4: Feature Importance Bar Chart

### 1. What & Why
Feature importance visualizations explain model decisions by ranking predictor features based on their contribution to impurity reduction (Decision Tree) or coefficient magnitude (Logistic Regression).

### 2. Code
```python
import numpy as np

def plot_feature_importances(importances, feature_names, model_name: str, top_n: int = 10):
    """Plot top N feature importances."""
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(8, 5))
    plt.barh(range(top_n), importances[indices][::-1], align='center', color='#2ecc71')
    plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Feature Importances — {model_name}')
    plt.tight_layout()
    plt.show()
```

### 3. Expected Output & How to Read It
* Bar chart listing top features top-to-bottom.
* On this dataset, `PageValues` typically dominates feature importance (> 50% relative weight).

### 4. Common Pitfalls
* Comparing raw Gini impurity importances directly with linear regression coefficients without scale standardization.

### 5. Git Checkpoint
* **Branch:** `ml/feature-eda`
* **Commit Message:** `feat(eda): add feature importance plotting utility`

---

## Tool 5: 2D Decision Boundary Map (KNN / SVM Intuition)

### 1. What & Why
Visualizing 2D decision boundaries on two selected features (`PageValues` vs `BounceRates`) builds intuition on how non-linear models (RBF SVM, KNN) partition feature space versus linear models (Logistic Regression).

### 2. Code
```python
def plot_2d_decision_boundary(model, X_2d, y, feature_cols: list, model_name: str):
    """Plot 2D decision boundary on 2 scaled numerical features."""
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='coolwarm', edgecolors='k', alpha=0.6, s=20)
    plt.xlabel(feature_cols[0])
    plt.ylabel(feature_cols[1])
    plt.title(f'2D Decision Boundary — {model_name}')
    plt.tight_layout()
    plt.show()
```

### 3. Expected Output & How to Read It
* Background shading shows decision regions (red = predict No Purchase, blue = predict Purchase).
* Linear models generate straight boundary lines; RBF SVM generates curved isolines; KNN displays irregular boundary islands.

### 4. Common Pitfalls
* Attempting to pass 28 features into a 2D plot function without fitting a dedicated 2-feature subset model.

### 5. Git Checkpoint
* **Branch:** `ml/feature-eda`
* **Commit Message:** `feat(eda): add 2D decision boundary visualization function`

---

## Tool 6: 5-Model Comparative Metric Bar Charts

### 1. What & Why
A grouped side-by-side bar chart summarizes final comparative performance across all 5 algorithms on accuracy, precision, recall, F1, and latency.

### 2. Code
```python
import pandas as pd

def plot_model_comparison(metrics_df: pd.DataFrame):
    """Plot grouped metric comparison bar chart across 5 algorithms."""
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    ax = metrics_df[metrics_to_plot].plot(kind='bar', figsize=(10, 6), colormap='Set2', width=0.8)
    plt.title("Algorithm Arena: 5-Model Metric Benchmark Comparison", fontsize=14)
    plt.ylabel("Score (0.0 to 1.0)")
    plt.xlabel("Model Architecture")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
```

### 3. Expected Output & How to Read It
* Grouped bar plot allowing instant visual evaluation of trade-offs (e.g. SVM high accuracy vs KNN high latency).

### 4. Common Pitfalls
* Mixing latency (milliseconds) on the same y-axis scale as normalized metric scores (0.0 to 1.0).

### 5. Git Checkpoint
* **Branch:** `ml/feature-eda`
* **Commit Message:** `feat(eda): add 5-model comparative bar chart plotting utility`
