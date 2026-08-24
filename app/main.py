from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI


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
def predict():
    features = [[6.0, 3.0, 4.8, 1.8]]

    prediction = app.state.model.predict(features)

    return {"prediction": int(prediction[0])}