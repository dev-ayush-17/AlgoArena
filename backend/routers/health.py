from fastapi import APIRouter, Response, status
from backend.config import settings
from backend.services.model_service import model_container

router = APIRouter(tags=["Health"])

@router.get("/health")
def health_check(response: Response):
    loaded_models = list(model_container.models.keys())
    is_healthy = model_container.is_loaded and len(loaded_models) > 0

    if not is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "unhealthy",
            "models_loaded": len(loaded_models),
            "error": "Model container not fully initialized."
        }
        
    return {
        "status": "healthy",
        "models_loaded": len(loaded_models),
        "models": loaded_models,
        "transformers_loaded": model_container.scaler is not None and model_container.encoder is not None
    }