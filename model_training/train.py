import os, joblib, time, sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from preprocessing import load_shoppers_data, preprocess_and_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# defining globally needed artifacts
# model evaluation and comparison file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
import model_comp_eval, verification

#metrics calculation
def calculate_metrics(y_test, y_pred_model, y_prob, pos_label_model=1, ):
    """
    Returns a dictionary of metrics to be passed to the tracker
    """
    return {
        "acc" : accuracy_score(y_test, y_pred_model),
        "prec" : precision_score(y_test, y_pred_model, pos_label=pos_label_model),
        "rec" : recall_score(y_test, y_pred_model, pos_label=pos_label_model),
        "f1" : f1_score(y_test, y_pred_model, pos_label=pos_label_model),
        "auc" : roc_auc_score(y_test, y_prob)
    }

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}\n")
    print(f"ROC-AUC-score: {auc:.4f}\n")
    print(classification_report(y_test, y_pred_model, target_names=['No Purchase (0)', 'Purchase (1)']))

# BENCHMARK LATENCY 
def benchmark_latency(model, sample_row: pd.DataFrame, warm_up: int=100, iterations: int = 1000) -> float:
    for _ in range(warm_up):
        _ = model.predict(sample_row)

    latencies_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = model.predict(sample_row)
        end = time.perf_counter()
        latencies_ms.append((end-start)*1000.0)
    return float(np.mean(latencies_ms))

# training models and logging
def train_and_log_model(model, model_key, display_name, x_train, x_test, y_train, y_test):
    """
    Helper function to execute standard training, eval and logging for any model.
    """
    print(f"\n---Training {display_name}")
    start_train = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - start_train

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(x_test))

    m = calculate_metrics(y_test, y_pred, y_prob)
    sample_row = x_test.iloc[[0]]
    latency = benchmark_latency(model, sample_row)

    model_path = os.path.join(ARTIFACTS_DIR, f"{model_key}.pkl")
    joblib.dump(model, model_path)
    file_size = os.path.getsize(model_path) / 1024.0

    model_comp_eval.metrics_per_model(
        model_key, display_name, m["acc"],m["prec"], m["rec"], m["f1"], m["auc"], 
        latency, file_size, train_time
    )
    print(f"Finished {display_name} | F1: {m['f1']:.4f} | Latency: {latency:.4f}ms" )


if __name__ == "__main__":
    raw_df = load_shoppers_data()
    x_train, x_test, y_train, y_test = preprocess_and_split(raw_df)

    # Trainig the required models
    models_to_train = [
        (LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42), 
         "logistic_regression", "Logistic Regression"),
         
        (KNeighborsClassifier(n_neighbors=5, weights='uniform', n_jobs=-1), 
         "knn", "K-Nearest Neighbors"),
         
        (SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced', random_state=42), 
         "svm", "Support Vector Machine"),
         
        (GaussianNB(var_smoothing=1e-9), 
         "naive_bayes", "Naive Bayes")
    ]

    for model, key, name in models_to_train:
        train_and_log_model(model, key, name, x_train, x_test, y_train, y_test)

    model_comp_eval.summarising_metrics()
    model_comp_eval.export_metrics(x_train, x_test)
    model_comp_eval.export_features()
    verification.verify_artifacts(ARTIFACTS_DIR)
    