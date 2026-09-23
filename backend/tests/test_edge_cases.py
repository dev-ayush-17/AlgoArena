def test_missing_features_payload_returns_422(client):
    # Missing required 'features' object
    response = client.post("/api/v1/predict", json={})
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "validation_error"

def test_invalid_type_field_returns_422(client):
    # Passing string for integer feature 'Administrative'
    payload = {
        "features": {
            "Administrative": "not_a_number"
        }
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "validation_error"

def test_out_of_range_bounce_rates_returns_400(client, valid_shopper_payload):
    # BounceRates > 1.0 violates physical ratio bounds
    payload = valid_shopper_payload.copy()
    payload["features"]["BounceRates"] = 5.0
    
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "invalid_feature_value"
    assert "BounceRates" in data["message"]

def test_invalid_month_category_returns_422(client, valid_shopper_payload):
    # Invalid Month category 'December' (dataset expects 'Dec')
    payload = valid_shopper_payload.copy()
    payload["features"]["Month"] = "December"
    
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "validation_error"