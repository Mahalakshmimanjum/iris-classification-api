# Iris Classification API

A Machine Learning API that predicts the species of an iris flower from its measurements.

The API is built using **FastAPI** and a **Scikit-learn Random Forest** model. It also includes API key authentication, input validation, API versioning, testing, logging, monitoring, and model explainability.

The application can be run locally or using Docker Compose.

---

## Project Goal

The goal of this project is to build a complete and production-style Machine Learning API.

The API takes four iris flower measurements:

* Sepal length
* Sepal width
* Petal length
* Petal width

It predicts one of three iris species:

* Setosa
* Versicolor
* Virginica

---

## Machine Learning Model

The project uses the **Iris dataset** and a **Random Forest Classifier** from Scikit-learn.

The trained model is stored at:

```text
ml/saved_model/model.joblib
```

The model receives four features and returns the predicted class.

Example input:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

---

## Architecture

```text
Client
  |
  v
FastAPI
  |
  v
API Key Authentication
  |
  v
Input Validation
  |
  v
Prediction Logic
  |
  v
Random Forest Model
  |
  v
JSON Response
```

---

## API Endpoints

### Health Check

```text
GET /api/v1/health
```

Checks whether the API and model are available.

Example:

```bash
curl http://localhost:8000/api/v1/health
```

---

### V1 Prediction

```text
POST /api/v1/predict
```

Predicts the iris species.

Example:

```bash
curl -X POST "http://localhost:8000/api/v1/predict" ^
-H "Content-Type: application/json" ^
-H "X-API-Key: your-secret-api-key" ^
-d "{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2}"
```

Example response:

```json
{
  "prediction": 0,
  "confidence": 1,
  "model_version": "1.0",
  "request_id": "063d772b-16f9-4c38-93af-86ef5284df10"
}
```

---

### V2 Prediction

```text
POST /api/v2/predict
```

V2 provides an updated response format that includes the probability distribution for the prediction.

Example:

```bash
curl -X POST "http://localhost:8000/api/v2/predict" ^
-H "Content-Type: application/json" ^
-H "X-API-Key: your-secret-api-key" ^
-d "{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2}"
```

---

### Batch Prediction

```text
POST /api/v1/predict-batch
```

Allows multiple iris flower measurements to be predicted in one request.

Example:

```bash
curl -X POST "http://localhost:8000/api/v1/predict-batch" ^
-H "Content-Type: application/json" ^
-H "X-API-Key: your-secret-api-key" ^
-d "[{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2},{\"sepal_length\":6.2,\"sepal_width\":2.8,\"petal_length\":4.8,\"petal_width\":1.8}]"
```

The maximum batch size is controlled by:

```text
MAX_BATCH_SIZE
```

---

### Model Information

```text
GET /api/v1/model-info
```

Returns information about the loaded machine learning model.

Example:

```bash
curl -H "X-API-Key: your-secret-api-key" http://localhost:8000/api/v1/model-info
```

---

### Prometheus Metrics

```text
GET /metrics
```

Provides application metrics that can be collected by Prometheus.

Example:

```bash
curl http://localhost:8000/metrics
```

The metrics include request information and prediction metrics.

---

### Swagger Documentation

FastAPI provides interactive API documentation at:

```text
http://localhost:8000/docs
```

Swagger UI can be used to test the API endpoints directly.

---

## API Security

Protected endpoints require an API key.

The API key is sent using the:

```text
X-API-Key
```

header.

Example:

```text
X-API-Key: your-secret-api-key
```

If the API key is missing or invalid, the API returns:

```json
{
  "detail": "Invalid or missing API key"
}
```

with HTTP status:

```text
401 Unauthorized
```

The API key is stored in the `.env` file and is not hardcoded in the application code.

---

## Input Validation

The API uses Pydantic for request validation.

The prediction input must contain:

```text
sepal_length
sepal_width
petal_length
petal_width
```

Validation includes:

* Required fields
* Numeric values
* Values greater than zero
* No unexpected fields

Invalid requests return:

```text
422 Unprocessable Entity
```

For example, an unexpected field is rejected:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2,
  "unexpected_field": "not_allowed"
}
```

---

## CORS

CORS is configured using allowed origins from the environment.

Example:

```env
ALLOWED_ORIGINS=http://localhost:3000
```

This avoids allowing every browser origin.

---

## Model Explainability

The project includes model explainability using:

* Random Forest Feature Importance
* LIME

Explainability files are stored in:

```text
explainability/
```

Main files include:

```text
explainability/
├── explain_model.py
├── feature_importance.png
├── lime_sample_1.html
└── lime_sample_2.html
```

These files help understand how the model makes predictions.

---

## Prediction Data Validation

The project also contains reusable prediction validation logic in:

```text
app/prediction.py
```

The function checks:

* Required fields
* Numeric values
* Finite values
* Positive values

Invalid data produces a clear error instead of sending incorrect data to the ML model.

---

## Testing

The project uses **Pytest** for automated testing.

Run the tests using:

```bash
python -m pytest -v
```

The test suite covers:

* Health endpoint
* Valid prediction
* Missing fields
* Invalid values
* Batch prediction
* Batch size limit
* Model information
* V1 and V2 response differences
* Missing API key
* Invalid API key
* Unexpected fields
* Prediction validation
* Wrong data types
* Negative values

Current test suite:

```text
15 tests passed
```

---

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
├── .env.example
└── README.md
```

---

# How to Run the Project

## Prerequisites

Install:

* Python 3.13
* Docker Desktop
* Docker Compose

---

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

Use your own secure value for `API_KEY`.

Do not commit the `.env` file to GitHub.

---

# Run with Docker Compose

Build and start the application:

```bash
docker compose up --build
```

The API will be available at:

```text
http://localhost:8000
```

Open Swagger:

```text
http://localhost:8000/docs
```

Open metrics:

```text
http://localhost:8000/metrics
```

---

## Stop the Application

```bash
docker compose down
```

---

# Run Tests Locally

Activate the virtual environment and run:

```bash
python -m pytest -v
```

You can also check installed dependencies:

```bash
pip check
```

Expected result:

```text
No broken requirements found.
```

---

# Docker

The application is containerized using Docker.

Basic flow:

```text
Python Base Image
       ↓
Install Dependencies
       ↓
Copy Application
       ↓
Load ML Model
       ↓
Start Uvicorn
       ↓
FastAPI Application
```

The application runs on port:

```text
8000
```

Docker Compose is used to make the project easier to run.

---

# Monitoring

The project includes Prometheus monitoring using:

```text
prometheus-fastapi-instrumentator
prometheus-client
```

Metrics are available at:

```text
http://localhost:8000/metrics
```

The metrics can be collected by Prometheus for monitoring API requests, response information, and prediction activity.

---

# Logging

The application includes logging using Python's logging module.

Logs help track:

* API requests
* Application events
* Errors
* Model loading

Rotating log files are used to avoid unlimited log file growth.

---

# API Versioning

The project supports two API versions.

### V1

```text
/api/v1/predict
```

Returns the original prediction response.

### V2

```text
/api/v2/predict
```

Returns an updated response that includes probability information.

This demonstrates how API versioning can be used when the response format changes.

---

# Technologies Used

* Python
* FastAPI
* Uvicorn
* Pydantic
* Pydantic Settings
* NumPy
* Scikit-learn
* Joblib
* LIME
* Matplotlib
* Pytest
* HTTPX
* Prometheus
* Docker
* Docker Compose

---

# What I Learned

During this project, I learned how to:

* Build a Machine Learning API using FastAPI
* Load and use a trained Scikit-learn model
* Create API request and response schemas
* Validate user input using Pydantic
* Protect API endpoints using API key authentication
* Create API versions using FastAPI routers
* Create batch prediction endpoints
* Handle API errors and exceptions
* Add logging to a FastAPI application
* Write automated tests using Pytest
* Add Prometheus monitoring
* Add model explainability using LIME
* Containerize an application using Docker
* Run the application using Docker Compose
* Manage application configuration using environment variables
* Maintain project dependencies using `requirements.txt`

---

# Project Status

The project currently includes:

* FastAPI ML API
* Random Forest Iris model
* V1 and V2 API endpoints
* Batch prediction
* API key authentication
* Input validation
* Error handling
* Logging
* Automated testing
* Prometheus metrics
* Model explainability
* Docker
* Docker Compose
* Project documentation

The project is ready for the final deployment and completion steps.
