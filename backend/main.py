from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.routers import health, predict, metrics, schema
from contextlib import asynccontextmanager
from backend.services.model_service import model_container
from fastapi.exceptions import RequestValidationError
from backend.routers.error_handlers import (
    MLValidationException, ModelInferenceException,
    validation_exception_handler, ml_validation_exception_handler,
    model_inference_exception_handler
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Initializing ML Artifact Container...")
    try:
        model_container.load_artifacts()
    except Exception as e:
        print(f"[FATAL] Failed to load ML artifacts: {e}")
        raise e
    yield
    print("[SHUTDOWN] Cleaning up resources...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(health.router, prefix=settings.API_V1_STR)
app.include_router(predict.router, prefix=settings.API_V1_STR)
app.include_router(metrics.router, prefix=settings.API_V1_STR)
app.include_router(schema.router, prefix=settings.API_V1_STR)

app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(MLValidationException, ml_validation_exception_handler)
app.add_exception_handler(ModelInferenceException, model_inference_exception_handler)

@app.get("/")
def root():
    return {
        "message": "Welcome to Algorithm Arena API. Visit /docs for OpenAPI specifications"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port="8000", reload=True)