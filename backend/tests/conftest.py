import pytest
import copy
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

@pytest.fixture
def valid_shopper_payload():
    return {
        "features": {
            "Administrative": 3,
            "Administrative_Duration": 87.0,
            "Informational": 1,
            "Informational_Duration": 12.0,
            "ProductRelated": 22,
            "ProductRelated_Duration": 640.5,
            "BounceRates": 0.0,
            "ExitRates": 0.015,
            "PageValues": 18.5,
            "SpecialDay": 0.0,
            "Month": "Nov",
            "OperatingSystems": 2,
            "Browser": 2,
            "Region": 1,
            "TrafficType": 2,
            "VisitorType": "Returning_Visitor",
            "Weekend": 0
        }
    }