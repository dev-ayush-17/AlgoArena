import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

PREPROCESSING_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PREPROCESSING_DIR)

# Constants & Paths
DATA_DIR = os.path.join(PREPROCESSING_DIR, "data", "raw")
CSV_PATH = os.path.join(DATA_DIR, "online_shoppers_intention.csv")
ARTIFACTS_DIR = os.path.join(PREPROCESSING_DIR, "artifacts")

# Feature Column Definitions
NUM_COLS = [
    'Administrative', 'Administrative_Duration', 'Informational', 
    'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
]
INT_CAT_COLS = ['OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Weekend']
OHE_CAT_COLS = ['Month', 'VisitorType']


def load_shoppers_data() -> pd.DataFrame:
    """Fetch dataset from UCI ML Repository or fallback to local CSV storage."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(CSV_PATH):
        print(f"[INFO] Loading dataset from local cache: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
    else:
        print("[INFO] Fetching dataset from UCI ML Repository (ID 468)...")
        dataset = fetch_ucirepo(id=468)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
        df.to_csv(CSV_PATH, index=False)
        print(f"[INFO] Saved local dataset copy to {CSV_PATH}")
    return df


def preprocess_and_split(
    df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Executes categorical encoding, train-test splitting, feature scaling,
    and saves fitted scaler/encoder artifacts to disk.
    """
    df = df.copy()
    
    # 1. Convert boolean flags to standard integer binary (0/1)
    df['Weekend'] = df['Weekend'].astype(int)
    df['Revenue'] = df['Revenue'].astype(int)

    # 2. Fit and transform categorical variables via OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(df[OHE_CAT_COLS])
    encoded_feature_names = encoder.get_feature_names_out(OHE_CAT_COLS)
    df_encoded = pd.DataFrame(encoded_cats, columns=encoded_feature_names, index=df.index)

    # 3. Construct unified feature set (X) and target (y)
    feature_cols = NUM_COLS + INT_CAT_COLS
    X = pd.concat([df[feature_cols], df_encoded], axis=1)
    y = df['Revenue']

    # 4. Stratified Train-Test Split (prevents data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 5. Fit StandardScaler ONLY on X_train numerical features
    scaler = StandardScaler()
    X_train_scaled_num = scaler.fit_transform(X_train[NUM_COLS])
    X_test_scaled_num = scaler.transform(X_test[NUM_COLS])

    # Assign scaled values back to working DataFrames
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    X_train_processed[NUM_COLS] = X_train_scaled_num
    X_test_processed[NUM_COLS] = X_test_scaled_num

    # 6. Save transformation artifacts for future inference/deployment
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    encoder_path = os.path.join(ARTIFACTS_DIR, "encoder.pkl")
    
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoder, encoder_path)
    print(f"[INFO] Saved fitted scaler to {scaler_path}")
    print(f"[INFO] Saved fitted encoder to {encoder_path}")

    return X_train_processed, X_test_processed, y_train, y_test


if __name__ == "__main__":
    # Execution entry point when running `python model_training/preprocessing.py`
    raw_df = load_shoppers_data()
    X_train, X_test, y_train, y_test = preprocess_and_split(raw_df)
    
    print("\n=== PREPROCESSING SUMMARY ===")
    print(f"X_train shape: {X_train.shape} | Positive Class Ratio: {y_train.mean():.4f}")
    print(f"X_test shape:  {X_test.shape} | Positive Class Ratio: {y_test.mean():.4f}")