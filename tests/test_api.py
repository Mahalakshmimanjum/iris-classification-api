def test_health(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_valid_input(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post(
        "/api/v1/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1, 2]
    assert 0 <= data["confidence"] <= 1
    assert data["model_version"] == "1.0"
    assert "request_id" in data


def test_predict_missing_field(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
    }

    response = client.post(
        "/api/v1/predict",
        json=payload
    )

    assert response.status_code == 422


def test_predict_invalid_value(client):
    payload = {
        "sepal_length": -5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post(
        "/api/v1/predict",
        json=payload
    )

    assert response.status_code == 422


def test_predict_batch_valid(client):
    payload = {
        "inputs": [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            },
            {
                "sepal_length": 6.2,
                "sepal_width": 3.4,
                "petal_length": 5.4,
                "petal_width": 2.3
            }
        ]
    }

    response = client.post(
        "/api/v1/predict-batch",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == 2


def test_predict_batch_oversized(client):
    single_input = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    payload = {
        "inputs": [single_input] * 101
    }

    response = client.post(
        "/api/v1/predict-batch",
        json=payload
    )

    assert response.status_code == 400

    data = response.json()

    assert "Batch size cannot exceed" in data["detail"]


def test_model_info(client):
    response = client.get("/api/v1/model-info")

    assert response.status_code == 200

    data = response.json()

    assert "model_type" in data
    assert "model_version" in data
    assert "training_date" in data
    assert "features" in data

    assert data["model_version"] == "1.0"


def test_v1_and_v2_have_different_response_shapes(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    v1_response = client.post(
        "/api/v1/predict",
        json=payload
    )

    v2_response = client.post(
        "/api/v2/predict",
        json=payload
    )

    assert v1_response.status_code == 200
    assert v2_response.status_code == 200

    v1_data = v1_response.json()
    v2_data = v2_response.json()

    # Both versions should return a valid prediction
    assert v1_data["prediction"] in [0, 1, 2]
    assert v2_data["prediction"] in [0, 1, 2]

    # v1 uses confidence
    assert "confidence" in v1_data
    assert "probabilities" not in v1_data

    # v2 uses full probability distribution
    assert "probabilities" in v2_data
    assert "confidence" not in v2_data

    # Version identifiers are different
    assert v1_data["model_version"] == "1.0"
    assert v2_data["model_version"] == "2.0"

    # Prove the response shapes are different
    assert set(v1_data.keys()) != set(v2_data.keys())