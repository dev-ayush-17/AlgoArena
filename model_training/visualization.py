import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score


# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, model_name: str):
    """
    Plots formatted confusion matrix with raw counts and normalised ratios
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['No purchase', 'Purchase'], yticklabels=['No purchase', 'Purchase'])


    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i+0.7, f"({cm_norm[i, j]*100:.1f}%)", ha="center", va="center", color="red" if cm[i,j] < 500 else "black")

    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()


# ROC Curve
def plot_roc_curve(y_true, y_probs, model_name: str):
    """
    Plots ROC curve and calculates AUC score
    """

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#2980b9', lw=2, label=f'{model_name} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate (1-Specificity)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Precision Recall Curve
def plot_precision_recall_curve(y_true, y_probs, model_name: str):
    """
    Plots Precision-Recall curve and calculates Average Precision score
    """

    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)
    baseline = (y_true == 1).sum() / len(y_true)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#e67e22', lw=2, label=f'{model_name} (AP = {avg_precision:.3f})')
    plt.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline ({baseline:.3f})')
    plt.xlabel('Recall Sensitivity')
    plt.ylabel('Precision Positive Predictive Value')
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Important Features
def plot_important_features(importances, features_names, model_name: str, top_n: int = 10):
    """
    Plots the top N important features based on the model
    """

    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(8,6))
    plt.barh(range(top_n), importances[indices][::-1], align='center', color='#2ecc71')
    plt.yticks(range(top_n), [features_names[i] for i in indices][::-1])
    plt.xlabel('Important Score')
    plt.title(f'Top {top_n} Feature Importances - {model_name}')
    plt.tight_layout()
    plt.show()


# 2D Decision Boundary
def plot_2d_decision_boundaries(model, x_2d, y, feature_cols: list, model_name: str):
    """
    Plot 2D decision boundary on 2 scaled numerical features
    """

    x_min, x_max = x_2d[:, 0].min() -1, x_2d[:,0].max() + 1
    y_min, y_max = x_2d[:, 1].min() -1, x_2d[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    plt.figure(figsize=(7,5))
    plt.contourf(xx, yy, z, alpha=0.3, cmap='coolwarm')
    plt.scatter(x_2d[:, 0], x_2d[:, 1], c=y, cmap='coolwarm', edgecolors='k', alpha=0.6, s=20)
    plt.xlabel(feature_cols[0])
    plt.ylabel(feature_cols[1])
    plt.title(f"2D Decision Boundary - {model_name}")
    plt.tight_layout()
    plt.show()


# Model Comparison
def plot_model_comparison(metrics_df: pd.DataFrame):
    """
    Plot grouped metric comparison bar chart across 5 Algorithms
    """
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    ax = metrics_df[metrics_to_plot].plot(kind='bar', figsize=(10, 6), colormap='Set2', width=0.8)
    plt.title("Algorithm Arena: 5 Model Metric Benchmark Comparison", fontsize=18)
    plt.ylabel("Score (0.0 to 1.0)")
    plt.xlabel("Model Architecture")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()