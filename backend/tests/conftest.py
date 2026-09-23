import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.model_service import model_container

@pytest.fixture(scope="session", autouse=True)
def initialize_artifacts():
    """Ensure artifacts are loaded once before running test session."""
    if not model_container.is_loaded:
        model_container.load_artifacts()

@pytest.fixture(scope="module")
def client():
    """Provides a reusable FastAPI TestClient instance."""
    with TestClient(app) as c:
        yield c