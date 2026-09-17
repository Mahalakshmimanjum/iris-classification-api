# Iris Classification API

A Machine Learning REST API that predicts the species of an iris flower from its measurements.

The API is built using **FastAPI** and a **Scikit-learn Random Forest Classifier**.

## Features

* Iris flower prediction
* FastAPI REST API
* API key authentication
* Input validation
* V1 and V2 API versions
* Batch prediction
* Automated testing with Pytest
* Prometheus monitoring
* Model explainability using LIME
* Logging
* Docker and Docker Compose
* GitHub Actions CI

---

## Machine Learning Model

The project uses the **Iris dataset** and a **Random Forest Classifier**.

The trained model is stored at:

```text
ml/saved_model/model.joblib
```

The model uses four inputs:

* Sepal length
* Sepal width
* Petal length
* Petal width

It predicts:

```text
0 = Setosa
1 = Versicolor
2 = Virginica
```

---

## Architecture

```text
Client
  |
  v
FastAPI
  |
  +--> API Key Authentication
  |
  +--> Input Validation
  |
  v
Prediction Logic
  |
  v
Random Forest Model
  |
  v
JSON Response

Monitoring --> Prometheus Metrics
Testing -----> Pytest
Deployment --> Docker / Docker Compose
```

---

## API Endpoints

| Method | Endpoint                | Description                   |
| ------ | ----------------------- | ----------------------------- |
| GET    | `/api/v1/health`        | Check API and model status    |
| POST   | `/api/v1/predict`       | Predict one flower            |
| POST   | `/api/v1/predict-batch` | Predict multiple flowers      |
| GET    | `/api/v1/model-info`    | Get model information         |
| POST   | `/api/v2/predict`       | Prediction with probabilities |
| GET    | `/metrics`              | Prometheus metrics            |
| GET    | `/docs`                 | Swagger documentation         |

Protected endpoints require the `X-API-Key` header.

---

## Example Prediction

### Request

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### V1 Request

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
  "request_id": "example-request-id"
}
```

### V2 Request

```bash
curl -X POST "http://localhost:8000/api/v2/predict" ^
-H "Content-Type: application/json" ^
-H "X-API-Key: your-secret-api-key" ^
-d "{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2}"
```

V2 also returns the probability information for each class.

---

## API Security

Protected endpoints use API key authentication.

Header:

```text
X-API-Key: your-secret-api-key
```

If the key is missing or incorrect, the API returns:

```text
401 Unauthorized
```

The API key is stored in the `.env` file.

---

## Input Validation

The API uses **Pydantic** for validation.

It checks:

* Required fields
* Numeric values
* Positive values
* Invalid data types
* Unexpected fields

Invalid input returns:

```text
422 Unprocessable Entity
```

---

## Run with Docker Compose

### 1. Create `.env`

Create a `.env` file in the project root:

```env
MODEL_PATH=ml/saved_model/model.joblib
LOG_LEVEL=INFO
MAX_BATCH_SIZE=100
API_TITLE=Iris Classification API
API_KEY=your-secret-api-key
ALLOWED_ORIGINS=http://localhost:3000
```

Do not commit the `.env` file to GitHub.

### 2. Start the application

```bash
docker compose up --build
```

The API will be available at:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

Metrics:

```text
http://localhost:8000/metrics
```

### 3. Stop the application

```bash
docker compose down
```

---

## Run Tests

Run the complete test suite:

```bash
python -m pytest -v
```

Current result:

```text
15 tests passed
```

Check dependencies:

```bash
pip check
```

Expected:

```text
No broken requirements found.
```

---

## Monitoring

Prometheus metrics are available at:

```text
http://localhost:8000/metrics
```

The project tracks:

* HTTP requests
* Request duration
* HTTP status codes
* ML prediction counts

---

## Model Explainability

The project uses:

* Random Forest Feature Importance
* LIME

Explainability files are stored in:

```text
explainability/
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
│   ├── models/
│   │   └── schemas.py
│   └── routers/
│       ├── v1.py
│       └── v2.py
│
├── ml/
│   └── saved_model/
│       └── model.joblib
│
├── explainability/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Technologies Used

* Python
* FastAPI
* Uvicorn
* Pydantic
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
* GitHub Actions

---

## What I Learned

Through this project, I learned how to:

* Build a Machine Learning API using FastAPI
* Use a trained Scikit-learn model
* Validate API input
* Add API key authentication
* Create API versions
* Build batch prediction APIs
* Write tests using Pytest
* Add Prometheus monitoring
* Add model explainability
* Use Docker and Docker Compose
* Manage configuration using environment variables
* Use GitHub Actions for automated testing

---

## Independent Extension

### GitHub Actions CI

GitHub Actions was added to automatically run the Pytest test suite when code is pushed to GitHub.

This helps find errors early and makes the project easier to maintain.

---

## Project Status

The project includes:

* Machine Learning API
* API security
* Input validation
* API versioning
* Batch prediction
* Automated testing
* Monitoring
* Model explainability
* Logging
* Docker
* Docker Compose
* GitHub Actions CI

The application can be run locally with:

```bash
docker compose up --build
```

The project is complete and ready for final review.
