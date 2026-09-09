import joblib
import pytest

from app.prediction import predict_new_customer


@pytest.fixture
def model():
    return joblib.load("ml/saved_model/model.joblib")


def test_valid_customer_prediction(model):
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    probability = predict_new_customer(data, model)

    assert 0.0 <= probability <= 1.0


def test_missing_required_field(model):
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
    }

    with pytest.raises(
        ValueError,
        match="Missing required fields: petal_width"
    ):
        predict_new_customer(data, model)


def test_wrong_data_type(model):
    data = {
        "sepal_length": "five",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    with pytest.raises(
        ValueError,
        match="sepal_length must be a number."
    ):
        predict_new_customer(data, model)


def test_negative_value(model):
    data = {
        "sepal_length": -5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    with pytest.raises(
        ValueError,
        match="sepal_length must be greater than 0."
    ):
        predict_new_customer(data, model)