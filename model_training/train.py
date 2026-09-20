import os, joblib, time, sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from preprocessing import load_shoppers_data, preprocess_and_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)



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



# training the Logistic Regression Model
def train_logreg(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test:pd.Series):
    """
    Trains, evaluate, benchmarks and serializes Logistic Regression model.
    """
    print("\n" + "="*40)
    print("      TRAINING LOGISTIC REGRESSION      ")
    print("="*40)

    logreg_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced',
        random_state=42
    )

    raw_df = load_shoppers_data()
    x_train_processed, x_test_processed, y_train, y_test = preprocess_and_split(raw_df)

    start_train = time.perf_counter()
    logreg_model.fit(x_train, y_train)
    train_time_sec = time.perf_counter() - start_train
    y_pred = logreg_model.predict(x_test)
    y_prob = logreg_model.predict_proba(x_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    auc = roc_auc_score(y_test, y_prob)

    print("\n===LOGISTIC REGRESSION EVALUATION===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=['No Purchase (0)', 'Purchase (1)']))

    sample_row = x_test.iloc[[0]]
    avg_latency = benchmark_latency(logreg_model, sample_row)
    print(f"[BENCHMARK] Single-Row Latency: {avg_latency:.4f} ms")

    model_path = os.path.join(ARTIFACTS_DIR, "logistic_regression.pkl")
    joblib.dump(logreg_model, model_path)
    file_size_kb = os.path.getsize(model_path) / 1024.0
    print(f"[INFO] Serialized model saved to: {model_path} ({file_size_kb:.2f} KB)")



# training the KNN Model
def train_KNN(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test:pd.Series):
    """
    Training of KNN algorithm on the given dataset 
    """
    knn_model = KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform',
        metric='minikowski',
        p=2,
        n_jobs=-1 
    )
    



if __name__ == "__main__":
    raw_df = load_shoppers_data()
    x_train, x_test, y_train, y_test = preprocess_and_split(raw_df)
    train_logreg(x_train, x_test, y_train, y_test)
