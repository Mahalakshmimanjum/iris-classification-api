from contextlib import asynccontextmanager
import uuid

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


@app.get("/health")
def health():
    model = getattr(app.state, "model", None)

    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(data: PredictionInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = app.state.model.predict(features)
    probabilities = app.state.model.predict_proba(features)

    confidence = max(probabilities[0])

    request_id = str(uuid.uuid4())

    return {
        "prediction": int(prediction[0]),
        "confidence": float(confidence),
        "request_id": request_id
    }