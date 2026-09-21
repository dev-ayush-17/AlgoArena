import os, sys
from pathlib import Path

def verify_artifacts(artifacts_dir):
    """
    Verifies if all artifacts are present in the artifacts directory before handling them to the backen API
    """   
    required_artifcats = [
        "logistic_regression.pkl",
        "knn.pkl",
        "svm.pkl",
        "naive_bayes.pkl",
        "scaler.pkl",
        "encoder.pkl",
        "feature_config.json",
        "metrics.json"
    ]

    print("=== FINAL ARTIFACT CHECKLIST VERIFICATION ===")

    missing_count = 0
    for artifact in required_artifcats:
        path = os.path.join(artifacts_dir, artifact)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024.0
            print(f" [x] {artifact:<25}({size_kb:>8.2f} KB)")
        else:
            print(f" [] {artifact:<25} MISSING!")
            missing_count += 1

    if missing_count == 0:
        print("\nALL 9 ARTIFACTS VERIFIED SUCCESSFULLY. Ready for Milestone 3 (Backend FastAPI)!")
    else:
        print(f"\nWARNING: {missing_count} required artifact(s) missing. Resolve before proceeding.")

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.append(SCRIPT_DIR)

    ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    verify_artifacts(ARTIFACTS_DIR)