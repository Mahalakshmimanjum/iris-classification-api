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

## Project Structure

```text
iris-classification-api/
│
├── app/
│   ├── main.py
│   ├── config.py
│   ├── exceptions.py
│   ├── logging_config.py
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
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── .env
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
```

Do not commit the `.env` file to GitHub because it may contain environment-specific or sensitive configuration.

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
pytest
```

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
* Scikit-learn
* NumPy
* Pandas
* SciPy
* Joblib
* LIME
* Pytest
* Docker
* Docker Compose
