def test_health_check_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] > 0
    assert "logistic_regression" in data["models"]
    assert data["transformers_loaded"] is True