import pytest

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

def test_predict_endpoint_happy_path(client, valid_shopper_payload):
    response = client.post("/api/v1/predict", json=valid_shopper_payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "predictions" in data
    assert "consensus" in data
    assert "input_echo" in data
    
    # Assert model list structure
    predictions = data["predictions"]
    assert len(predictions) >= 4
    
    for pred in predictions:
        assert pred["model_key"] in ["logistic_regression", "knn", "svm", "naive_bayes", "decision_tree"]
        assert pred["prediction"] in [0, 1]
        assert pred["prediction_label"] in ["No Purchase", "Purchase Completed"]
        assert 0.0 <= pred["confidence"] <= 1.0
        # Latency check: prediction should take under 500ms
        assert 0.0 <= pred["inference_time_ms"] < 500.0

    # Consensus checks
    consensus = data["consensus"]
    assert consensus["majority_prediction"] in [0, 1]
    assert 0.0 <= consensus["agreement_ratio"] <= 1.0