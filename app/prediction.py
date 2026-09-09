from typing import Any

import numpy as np


REQUIRED_FIELDS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


def predict_new_customer(
    customer_data: dict[str, Any],
    model
) -> float:
    """
    Validate new customer/flower data and return
    the probability of the predicted class.
    """

    # Check input type
    if not isinstance(customer_data, dict):
        raise ValueError("Input must be a dictionary.")

    # Check for missing fields
    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if field not in customer_data
    ]

    if missing_fields:
        raise ValueError(
            f"Missing required fields: {', '.join(missing_fields)}"
        )

    # Validate values
    features = []

    for field in REQUIRED_FIELDS:
        value = customer_data[field]

        if not isinstance(value, (int, float)):
            raise ValueError(
                f"{field} must be a number."
            )

        if not np.isfinite(value):
            raise ValueError(
                f"{field} must be a finite number."
            )

        if value <= 0:
            raise ValueError(
                f"{field} must be greater than 0."
            )

        features.append(float(value))

    try:
        prediction = model.predict([features])
        probabilities = model.predict_proba([features])

        predicted_class = int(prediction[0])
        probability = float(probabilities[0][predicted_class])

        return probability

    except Exception as e:
        raise RuntimeError(
            f"Prediction failed: {e}"
        ) from e