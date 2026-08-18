# Iris Classification API

## Project Goal

This project will build a simple Machine Learning API that predicts the species of an iris flower based on its measurements.

## Dataset

We are using the Iris dataset.

The dataset contains measurements of iris flowers and their species.

The input features are:
- Sepal length
- Sepal width
- Petal length
- Petal width

The possible flower species are:
- Setosa
- Versicolor
- Virginica

## ML Problem

This is a classification problem.

The model will learn from the Iris dataset and predict which species an iris flower belongs to.

## API Contract

The `/predict` endpoint will accept the four measurements of an iris flower: sepal length, sepal width, petal length, and petal width.

The API will validate the input values and send them to the trained machine learning model. The model will predict the flower species. The API will return the predicted species as a JSON response.

Example input:

{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

Example output:

{
  "species": "setosa"
}

## Request Flow

User Request
→ Input Validation
→ Machine Learning Model
→ Prediction
→ JSON Response

The API will receive the flower measurements, validate them, pass them to the ML model, and return the predicted iris species.