import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

logger = logging.getLogger("backend.error_handlers")

class MLValidatorException(Exception):
    """Raised when feature inputs break physical/domain boundaries."""
    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field = field

class ModelInferenceException(Exception):
    """Raised when a specific ML model fails during predict execution."""
    def __init__(self, message: str, model_key: str):
        self.message = message
        self.model_key = model_key


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Formats Pydantic 422 validation errors into standard project error shape."""
    details = []
    for error in exc.errors():
        loc = [str(x) for x in error.get("loc", [])]
        details.append({
            "loc": loc,
            "msg": error.get("msg", "Invalid value"),
            "type": error.get("type", "value_error")
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "message": "Input features failed payload validation.",
            "details": details
        }
    )

async def ml_validation_exception_handler(request: Request, exc: MLValidationException):
    """Formats 400 Bad Request domain boundary violations."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "invalid_feature_value",
            "message": exc.message,
            "field": exc.field
        }
    )

async def model_inference_exception_handler(request: Request, exc: ModelInferenceException):
    """Formats 500 Internal Server Error when ML inference execution fails."""
    logger.error(f"Inference error on model '{exc.model_key}': {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "prediction_error",
            "message": f"Model '{exc.model_key}' failed: {exc.message}",
            "failed_models": [exc.model_key]
        }
    )