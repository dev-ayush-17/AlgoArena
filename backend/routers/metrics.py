import json, logging
from fastapi import APIRouter, HTTPException, status
from backend.config import settings
from backend.schemas.metrics import MetricsResponse
from typing import Optional

router = APIRouter(tags=["Metrics"])
logger = logging.getLogger("backend.routers.metrics")

_metrics_cache: Optional[dict] = None

def load_metrics_json() -> dict:
    global _metrics_cache
    if _metrics_cache is not None:
        return _metrics_cache

    metrics_path = settings.ARTIFACTS_DIR / settings.METRICS_FILENAME
    if not metrics_path.exists():
        logger.error(f"Metrics.json missing at {metrics_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Benchmark metrics artifacts missing"
        )

    try:
        with open(metrics_path, "r") as f:
            data = json.load(f)

        models = data.get("models", {})
        if models:
            best_per_metric = {
                "highest_accuracy": max(models.items(), key= lambda x: x[1].get(
                    "accuracy", 0
                ))[0],
                "highest_f1": max(models.items(), key=lambda x: x[1].get(
                    "f1_score", 0
                ))[0],
                "fastest_inference": min(models.items(), key=lambda x: x[1].get(
                    "avg_inference_time_ms", float("inf")
                ))[0],
                "smallest_file": min(models.items(), key=lambda x: x[1].get(
                    "model_file_size_kb", float("inf")
                ))[0]
            }
            data["best_per_metric"] = best_per_metric

        _metrics_cache = data
        return data

    except Exception as e:
        logger.error(f"Error reading metrics.json: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read benchmark metrics: {str(e)}"
        )

@router.get("/metrics", response_model=MetricsResponse, status_code=status.HTTP_200_OK)
def get_model_benchmark_metrics():
    """
    Retrieves pre-computed evaluation metrics (Accuracy, Precision, Recall, F1, 
    Inference Latency, File Size) for all trained models.
    """
    return load_metrics_json()