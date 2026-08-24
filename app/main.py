from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI

from app.models.schemas import PredictionInput


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ML model...")

    app.state.model = joblib.load(
        "ml/saved_model/model.joblib"
    )

    print("ML model loaded successfully!")

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {"message": "ML API is alive"}


@app.post("/predict")
def predict(data: PredictionInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = app.state.model.predict(features)

    return {"prediction": int(prediction[0])}