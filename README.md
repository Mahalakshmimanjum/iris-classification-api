# Iris Classification API

## Project Goal

This project builds a Machine Learning API that predicts the species of an iris flower based on its measurements.

The API is developed using **FastAPI** and a trained **Scikit-learn Random Forest** machine learning model. The application is containerized using **Docker** and can be started using **Docker Compose**.

## Dataset

We are using the Iris dataset.

The dataset contains measurements of iris flowers and their species.

The input features are:

* Sepal length
* Sepal width
* Petal length
* Petal width

The possible flower species are:

* Setosa
* Versicolor
* Virginica

## ML Problem

This is a classification problem.

The model learns from the Iris dataset and predicts which species an iris flower belongs to.

The trained model is saved as:

```text
ml/saved_model/model.joblib
```

## API Contract

The API accepts the four measurements of an iris flower:

* `sepal_length`
* `sepal_width`
* `petal_length`
* `petal_width`

The API validates the input values and sends them to the trained machine learning model. The model then predicts the flower class and returns the result as a JSON response.

### Example Input

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Example Output — V1

```json
{
  "prediction": 0,
  "confidence": 1,
  "model_version": "1.0",
  "request_id": "063d772b-16f9-4c38-93af-86ef5284df10"
}
```

### API Versions

The project supports multiple API versions:

* `/api/v1/predict` — Version 1 prediction endpoint
* `/api/v2/predict` — Version 2 prediction endpoint with an updated response format
* `/api/v1/predict-batch` — Batch prediction endpoint
* `/docs` — Interactive Swagger API documentation

## Request Flow

```text
User Request
     ↓
API Key Authentication
     ↓
Input Validation
     ↓
Machine Learning Model
     ↓
Prediction
     ↓
JSON Response
```

The API receives the flower measurements, validates them, passes them to the ML model, and returns the predicted iris class.

## Model Explainability

The project also includes model explainability using:

* Random Forest Feature Importance
* LIME (Local Interpretable Model-Agnostic Explanations)

Explainability files are available in:

```text
explainability/
```

This helps understand which features contribute most to the model's predictions.

## Security and Robustness

### API Key Authentication

The API requires a valid `X-API-Key` header for protected requests.

The API key is stored in the `.env` file and loaded through application settings. It is not hardcoded in the application code.

Example request header:

```text
X-API-Key: your-secret-api-key
```

Requests with a missing or invalid API key are rejected with:

```json
{
  "detail": "Invalid or missing API key"
}
```

and HTTP status:

```text
401 Unauthorized
```

### Input Validation

The API uses Pydantic validation to reject malformed input.

The prediction input follows these validation rules:

* All four feature values must be greater than zero.
* Empty or invalid values are rejected.
* Unexpected fields are rejected using `extra="forbid"`.
* Invalid request data returns HTTP `422 Unprocessable Entity`.

For example, the following request is rejected because `unexpected_field` is not part of the schema:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2,
  "unexpected_field": "not_allowed"
}
```

### CORS Configuration

Cross-Origin Resource Sharing (CORS) is configured using explicitly allowed origins.

The allowed origins are stored in the `ALLOWED_ORIGINS` environment variable instead of allowing all origins.

Example:

```env
ALLOWED_ORIGINS=http://localhost:3000
```

This prevents unauthorized browser origins from accessing the API through cross-origin requests.

### Rate Limiting

Rate limiting is an important security mechanism used to prevent excessive requests and reduce API abuse.

This project documents the rate-limiting concept but does not implement a production-grade rate limiter.

For a production deployment, rate limiting could be implemented using:

* Redis-based rate limiting
* An API gateway
* Reverse proxy rate limiting
* A dedicated FastAPI rate-limiting library

A production rate limiter should limit requests based on factors such as API key, client IP, or user identity.

## Data Validation and Pipeline Robustness

The project includes a reusable `predict_new_customer()` function for making predictions on new and potentially messy input data.

The function is available in:

```text
app/prediction.py
```

### Prediction Validation

Before sending data to the machine learning model, the function validates:

* Required fields are present.
* Input values are numeric.
* Values are finite numbers.
* Feature values are greater than zero.
* Invalid input is rejected with a clear error message.

This prevents invalid data from reaching the ML model and reduces the risk of incorrect or meaningless predictions.

### Example Valid Input

```python
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
```

The function validates the input and returns the probability of the predicted class.

### Invalid Input Handling

The function safely handles different types of invalid data.

#### Missing Field

```text
ValueError: Missing required fields: petal_width
```

#### Wrong Data Type

```text
ValueError: sepal_length must be a number.
```

#### Negative Value

```text
ValueError: sepal_length must be greater than 0.
```

Instead of silently producing a prediction, the function raises a clear error explaining what is wrong with the input.

### Automated Testing

The robustness of the prediction function is verified using Pytest.

Tests cover:

* Valid new customer/flower prediction
* Missing required fields
* Wrong data types
* Negative feature values

The dedicated tests are located in:

```text
tests/test_prediction.py
```

All four dedicated robustness tests pass successfully.

## Project Structure

```text
iris-classification-api/
│
├── app/
│   ├── main.py
│   ├── config.py
│   ├── security.py
│   ├── prediction.py
│   ├── exceptions.py
│   ├── logging_config.py
│   │
│   ├── models/
│   │   └── schemas.py
│   │
│   └── routers/
│       ├── v1.py
│       └── v2.py
│
├── ml/
│   └── saved_model/
│       └── model.joblib
│
├── explainability/
│   ├── explain_model.py
│   ├── feature_importance.png
│   ├── lime_sample_1.html
│   └── lime_sample_2.html
│
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   └── test_prediction.py
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── .env
├── .env.example
└── README.md
```

# How to Run This Project

## Prerequisites

Make sure the following are installed:

* Python 3.13
* Docker Desktop
* Docker Compose

## Environment Variables

Create a `.env` file in the project root.

Example:

```env
MODEL_PATH=ml/saved_model/model.joblib
LOG_LEVEL=INFO
MAX_BATCH_SIZE=100
API_TITLE=Iris Classification API
API_KEY=your-secret-api-key
ALLOWED_ORIGINS=http://localhost:3000
```

The `API_KEY` should be replaced with a secure secret value.

Do not commit the `.env` file to GitHub because it may contain sensitive configuration.

## Run Using Docker Compose

Build the Docker image and start the API with:

```bash
docker compose up --build
```

Once the application starts successfully, the API will be available at:

```text
http://localhost:8000
```

### Swagger API Documentation

Open the following URL in your browser:

```text
http://localhost:8000/docs
```

From Swagger UI, you can test:

```text
/api/v1/predict
/api/v2/predict
/api/v1/predict-batch
```

For protected endpoints, provide the required `X-API-Key` header.

### Stop the Application

To stop the containers:

```bash
docker compose down
```

## Docker Compose Model Volume

Docker Compose uses a named volume for the saved machine learning model:

```yaml
volumes:
  - model_data:/app/ml/saved_model
```

This separates the model storage from the application container. A retrained model can be swapped into the model storage without rebuilding the entire application image.

## Run Tests

To run the automated tests locally:

```bash
python -m pytest
```

The test suite covers:

* Health endpoint
* Valid predictions
* Invalid input
* Missing input fields
* Batch predictions
* Batch size limits
* Model information
* V1 and V2 response differences
* Missing API key
* Invalid API key
* Unexpected extra fields
* Valid new data prediction
* Missing fields in new data
* Wrong data types
* Negative feature values

## Docker Image

The API is containerized using the following Docker image configuration:

```text
Python 3.13 Slim
        ↓
Install dependencies
        ↓
Copy application code
        ↓
Expose port 8000
        ↓
Start FastAPI using Uvicorn
```

The API listens on `0.0.0.0` inside the container so that it can accept connections through Docker's published port.

## Technologies Used

* Python 3.13
* FastAPI
* Uvicorn
* Pydantic
* Pydantic Settings
* Scikit-learn
* NumPy
* Pandas
* SciPy
* Joblib
* LIME
* Pytest
* Docker
* Docker Compose
