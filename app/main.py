from contextlib import asynccontextmanager
import uuid

import joblib
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.models.schemas import PredictionInput, PredictionOutput


class PredictionError(Exception):
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ML model...")

    app.state.model = joblib.load(
        "ml/saved_model/model.joblib"
    )

    print("ML model loaded successfully!")

    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(PredictionError)
async def prediction_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Prediction failed"}
    )


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


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    request_id = str(uuid.uuid4())

    try:
        prediction = app.state.model.predict(features)
        probabilities = app.state.model.predict_proba(features)

        confidence = max(probabilities[0])

    except Exception as e:
        print(f"Prediction error: {e}")
        raise PredictionError()

    return {
        "prediction": int(prediction[0]),
        "confidence": float(confidence),
        "model_version": "1.0",
        "request_id": request_id
    } 