from fastapi import APIRouter, HTTPException, status
from backend.schemas.predict import PredictionResponse, PredictionRequest
from backend.services.model_service import run_prediction_pipeline, model_container

router = APIRouter(tags=["Prediction"])

@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_shopper_intention(request: PredictionRequest):
    """
    Submits e-commerce session features to all 5 pre-trained models.
    
    Returns side-by-side predictions, confidence scores, inference latencies,
    and consensus summary.
    """
    if not model_container.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models are not loaded into server memory."
        )

    try:
        response = run_prediction_pipeline(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution error: {str(e)}"
        )