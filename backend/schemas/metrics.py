from pydantic import BaseModel
from typing import Dict, List, Any, Optional

class ModelMetricDetails(BaseModel):
    display_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_inference_time_ms: float
    model_file_size_kb: float
    training_time_seconds: float

class DatasetSizeInfo(BaseModel):
    total_rows: int
    train_rows: int
    test_rows: int
    features: int
    features_after_encoding: int

class MetricsResponse(BaseModel):
    generated_at: str
    dataset: str
    dataset_size: DatasetSizeInfo
    models: Dict[str, ModelMetricDetails]
    best_per_metric: Optional[Dict[str, str]] = None