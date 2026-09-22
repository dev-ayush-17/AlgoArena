from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.routers import health, predict
from contextlib import asynccontextmanager
from backend.services.model_service import model_container


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
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=["*"],
    allow_headers=["*"],
    allow_origins=["*"],
    allow_methods=["*"]
)

app.include_router(health.router, prefix=settings.API_V1_STR)
app.include_router(predict.router, prefix=settings.API_V1_STR)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(health.router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {
        "message": "Welcome to Algorithm Arena API. Visit /docs for OpenAPI specifications"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port="8000", reload=True)