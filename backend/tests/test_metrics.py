def test_get_metrics_endpoint(client):
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "dataset" in data
    assert "models" in data
    assert "logistic_regression" in data["models"]
    
    # Assert specific metric keys
    lr = data["models"]["logistic_regression"]
    assert 0.0 <= lr["accuracy"] <= 1.0
    assert 0.0 <= lr["f1_score"] <= 1.0
    assert lr["avg_inference_time_ms"] >= 0.0
    assert "best_per_metric" in data

def test_get_form_schema_endpoint(client):
    response = client.get("/api/v1/form-schema")
    assert response.status_code == 200
    data = response.json()
    assert "numerical_features" in data
    assert "categorical_features" in data
    assert len(data["numerical_features"]) == 10
    assert len(data["categorical_features"]) == 7