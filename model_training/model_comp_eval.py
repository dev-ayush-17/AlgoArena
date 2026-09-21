import pandas as pd
import os, json, sys
from datetime import datetime, timezone

metrics_summary = []
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def summarising_metrics():
    """Prints a clean DataFrame summary of all trained models in the terminal."""
    if not metrics_summary:
        print("No metrics to summarize.")
        return
    
    df_summary = pd.DataFrame(metrics_summary).set_index("display_name")
    print("\n" + "="*50)
    print("           ALGORITHM ARENA BENCHMARK SUMMARY")
    print("="*50)
    print(df_summary[['f1_score', 'roc_auc', 'precision', 'recall', 'avg_inference_time_ms', 'model_file_size_kb']])
    print("="*50 + "\n")

def metrics_per_model(model_name, display_name, acc, prec, rec, f1, auc, avg_latency, model_size, train_time):
    summary = {
        "model_key": model_name,
        "display_name": display_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "avg_inference_time_ms": round(avg_latency, 4),
        "model_file_size_kb": round(model_size, 2),
        "training_time_seconds": round(train_time, 4)
    }
    metrics_summary.append(summary)

def export_metrics(x_train, x_test):
    metrics_json_path = os.path.join(ARTIFACTS_DIR, "metrics.json")

    metrics_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "Online Shoppers Purchasing Intention Dataset (UCI ID 468)",
        "dataset_size": {
            "total_rows": 12330,
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "features": 17,
            "features_after_encoding": x_train.shape[1]
        },
        "models": {
            item["model_key"]: {
                "display_name": item["display_name"],
                "accuracy": item["accuracy"],
                "precision": item["precision"],
                "recall": item["recall"],
                "f1_score": item["f1_score"],
                "avg_inference_time_ms": item["avg_inference_time_ms"],
                "model_file_size_kb": item["model_file_size_kb"],
                "training_time_seconds": item["training_time_seconds"]
            } for item in metrics_summary
        }
    }

    with open(metrics_json_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"Successfully generated metrics.json at: {metrics_json_path}")

def export_features():
    feature_config_path = os.path.join(ARTIFACTS_DIR, "feature_config.json")

    feature_config = {
        "numerical_features": [
            {"name": "Administrative", "dtype": "int64", "min": 0, "max": 27, "default": 0, "description": "Number of administrative pages visited"},
            {"name": "Administrative_Duration", "dtype": "float64", "min": 0.0, "max": 3398.0, "default": 0.0, "description": "Total time spent on administrative pages (seconds)"},
            {"name": "Informational", "dtype": "int64", "min": 0, "max": 24, "default": 0, "description": "Number of informational pages visited"},
            {"name": "Informational_Duration", "dtype": "float64", "min": 0.0, "max": 2549.0, "default": 0.0, "description": "Total time spent on informational pages (seconds)"},
            {"name": "ProductRelated", "dtype": "int64", "min": 0, "max": 705, "default": 1, "description": "Number of product-related pages visited"},
            {"name": "ProductRelated_Duration", "dtype": "float64", "min": 0.0, "max": 63973.5, "default": 10.0, "description": "Total time spent on product-related pages (seconds)"},
            {"name": "BounceRates", "dtype": "float64", "min": 0.0, "max": 0.2, "default": 0.0, "description": "Percentage of visitors entering site and leaving immediately"},
            {"name": "ExitRates", "dtype": "float64", "min": 0.0, "max": 0.2, "default": 0.02, "description": "Percentage of pageviews that were last in session"},
            {"name": "PageValues", "dtype": "float64", "min": 0.0, "max": 361.0, "default": 0.0, "description": "Average Google Analytics value of visited pages"},
            {"name": "SpecialDay", "dtype": "float64", "min": 0.0, "max": 1.0, "default": 0.0, "description": "Closeness of site visit time to a special day (0.0 to 1.0)"}
        ],
        "categorical_features": [
            {"name": "Month", "dtype": "string", "categories": ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], "default": "May", "description": "Month of the session"},
            {"name": "OperatingSystems", "dtype": "int64", "categories": [1, 2, 3, 4, 5, 6, 7, 8], "default": 1, "description": "Operating system ID"},
            {"name": "Browser", "dtype": "int64", "categories": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "default": 2, "description": "Browser ID"},
            {"name": "Region", "dtype": "int64", "categories": [1, 2, 3, 4, 5, 6, 7, 8, 9], "default": 1, "description": "Geographic region ID"},
            {"name": "TrafficType", "dtype": "int64", "categories": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "default": 2, "description": "Traffic source type ID"},
            {"name": "VisitorType", "dtype": "string", "categories": ["Returning_Visitor", "New_Visitor", "Other"], "default": "Returning_Visitor", "description": "Visitor status"},
            {"name": "Weekend", "dtype": "int64", "categories": [0, 1], "default": 0, "description": "Session occurred on weekend (0=No, 1=Yes)"}
        ],
        "target": {
            "name": "Revenue",
            "classes": [0, 1],
            "class_labels": ["No Purchase", "Purchase Completed"]
        }
    }

    with open(feature_config_path, "w") as f:
        json.dump(feature_config_path, f, indent=2)

    print(f"Successfully generated feature comnfig.json at: {feature_config_path}")
