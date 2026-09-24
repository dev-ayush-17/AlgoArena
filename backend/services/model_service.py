import numpy as np
import pandas as pd
import os, joblib, logging
from typing import Dict, Any, List, Optional
from backend.services.preprocessing_services import transform_features_for_inference
from backend.schemas.predict import (
    PredictionRequest, PredictionResponse, SingleModelPrediction, ConsensusSummary
)
from backend.utils.timing import measure_inference_time_ms
from backend.config import settings

logger = logging.getLogger("backend.services.model_service")

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "knn": "K-Nearest Neighbors",
    "svm": "Support Vector Machine",
    "decision_tree": "Decision Tree",
    "naive_bayes": "Naive Bayes"
}

CLASS_LABELS = {
    0: "No Purchase",
    1: "Purchase Completed"
}

def run_prediction_pipeline(request: PredictionRequest) -> PredictionResponse:
    """
    Orchestrates input feature transformation, multi-model execution,
    latency benchmarking, and consensus aggregation.
    """
    x_processed = transform_features_for_inference(
        request.features, 
        scaler=model_container.scaler, 
        encoder=model_container.encoder
    )
    single_predictions: List[SingleModelPrediction] = []
    raw_preds: List[int] = []
    failed_models: List[str] = []

    for model_key, model in model_container.models.items():
        display_name = DISPLAY_NAMES.get(model_key, model_key.replace("_", " ").title())

        try:
            prediction_arr, latency_ms = measure_inference_time_ms(model.predict, x_processed)
            pred_int = int(prediction_arr[0])
            raw_preds.append(pred_int)

            confidence = 0.5
            if hasattr(model, "predict_proba"):
                try:
                    proba_arr = model.predict_proba(x_processed)[0]
                    confidence = float(np.max(proba_arr))
                except Exception:
                    confidence = 0.5

            single_predictions.append(
                SingleModelPrediction(
                    model_name=display_name,
                    model_key=model_key,
                    prediction=pred_int,
                    prediction_label=CLASS_LABELS.get(pred_int, f"Class {pred_int}"),
                    confidence=round(confidence, 4),
                    inference_time_ms=latency_ms
                )
            )
        except Exception as e:
            logger.error(f"Model '{model_key}' prediction failed: {str(e)}")
            failed_models.append(model_key)

    if not single_predictions:
        raise RuntimeError("All models failed during inference execution!")

    majority_pred = int(max(set(raw_preds), key=raw_preds.count))
    models_agreeing = raw_preds.count(majority_pred)
    models_total = len(raw_preds)
    agreement_ratio = round(models_agreeing / models_total, 4)

    consensus = ConsensusSummary(
        majority_prediction=majority_pred,
        majority_label=CLASS_LABELS.get(majority_pred, f"Class {majority_pred}"),
        agreement_ratio=agreement_ratio,
        models_agreeing=models_agreeing,
        models_total=models_total
    )

    response_dict = {
        "predictions": single_predictions,
        "consensus": consensus,
        "input_echo": {
            "Administrative": request.features.Administrative,
            "ProductRelated": request.features.ProductRelated,
            "PageValues": request.features.PageValues,
            "Month": request.features.Month,
            "VisitorType": request.features.VisitorType,
            "Weekend": request.features.Weekend
        }
    }
    
    if failed_models:
        response_dict["failed_models"] = failed_models

    return PredictionResponse(**response_dict)

class ModelContainer:
    """
    Singleton container holding all serialized models and preprocessors in memory
    """

    def __init__(self):
        self.scaler: Optional[Any] = None
        self.encoder: Optional[Any] = None
        self.models: Dict[str, Any] = {}
        self.is_loaded: bool = False

    def load_artifacts(self) -> None:
        """
        Loads fitted transformers and models in memory
        """
        artifacts_dir = settings.ARTIFACTS_DIR
        logger.info(f"Loading ML artifacts from {artifacts_dir}...")

        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Artifacts directory missing at {artifacts_dir}")

        scaler_path = artifacts_dir / settings.SCALER_FILENAME
        encoder_path = artifacts_dir / settings.ENCODER_FILENAME

        if not scaler_path.exists() or not encoder_path.exists():
            raise FileNotFoundError("scaler.pkl or encoder.pkl missing from artifacts directory!")

        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)
        logger.info(f"Successfully loaded scaler and encoder artifacts")

        loaded_count = 0
        for model_key, filename in settings.MODEL_FILES.items():
            model_path = artifacts_dir / filename
            if not model_path.exists():
                logger.warning(f"Model artifact missing for {model_key}: {model_path}. Skipping...")
                continue

            try:
                model = joblib.load(model_path)
                self.models[model_key] = model
                loaded_count += 1
                logger.info(f"Loaded model '{model_key}'")
            except Exception as e:
                logger.error(f"Failed to unpickle model {model_key}: {e}")

        if loaded_count == 0:
            raise RuntimeError("No model artifacts were successfully loaded into memory")

        self.is_loaded = True
        logger.info(f"Artifact initialization complete. Total models read = {loaded_count}")

model_container = ModelContainer()