"""
Tests for static frontend serving via FastAPI.
Verifies that index.html or static routes serve correctly when static output is present.
"""

from fastapi.testclient import TestClient
from backend.main import app
from pathlib import Path

client = TestClient(app)
frontend_dir = Path(__file__).parent.parent.parent / "frontend"
has_static_build = (frontend_dir / "out" / "index.html").exists() or (frontend_dir / "index.html").exists()

def test_serve_index_html():
    response = client.get("/")
    if has_static_build:
        assert response.status_code == 200
        assert "ALGORITHM ARENA" in response.text
    else:
        # Decoupled mode: Frontend runs via Next.js dev server
        assert response.status_code in [200, 404]

def test_health_check_available():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

