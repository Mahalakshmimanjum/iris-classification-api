import time

from fastapi import APIRouter, Request

from app.exceptions import PredictionError
from app.logging_config import setup_logging
from app.models.schemas import (
    PredictionInput,
    PredictionOutput,
    PredictionBatchInput,
    PredictionBatchOutput,
)

logger = setup_logging()

router = APIRouter(prefix="/api/v1")


@router.get("/health")
def health(request: Request):
    model = getattr(request.app.state, "model", None)

    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@router.post("/predict", response_model=PredictionOutput)
def predict(
    data: PredictionInput,
    request: Request
):
    request_id = request.state.request_id

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    try:
        model = request.app.state.model

        prediction = model.predict(features)
        probabilities = model.predict_proba(features)

        confidence = max(probabilities[0])

        logger.info(
            f"prediction_success "
            f"request_id={request_id} "
            f"prediction={int(prediction[0])}"
        )

    except Exception as e:
        logger.error(
            f"prediction_failed "
            f"request_id={request_id} "
            f"error={e}"
        )
        raise PredictionError()

    return {
        "prediction": int(prediction[0]),
        "confidence": float(confidence),
        "model_version": "1.0",
        "request_id": request_id
    }


@router.post(
    "/predict-batch",
    response_model=PredictionBatchOutput
)
def predict_batch(
    data: PredictionBatchInput,
    request: Request
):
    request_id = request.state.request_id
    start_time = time.perf_counter()

    try:
        model = request.app.state.model

        features = [
            [
                item.sepal_length,
                item.sepal_width,
                item.petal_length,
                item.petal_width
            ]
            for item in data.inputs
        ]

        predictions = model.predict(features)
        probabilities = model.predict_proba(features)

        results = []

        for prediction, probability in zip(
            predictions,
            probabilities
        ):
            results.append(
                PredictionOutput(
                    prediction=int(prediction),
                    confidence=float(max(probability)),
                    model_version="1.0",
                    request_id=request_id
                )
            )

        duration = time.perf_counter() - start_time

        logger.info(
            f"batch_prediction_success "
            f"request_id={request_id} "
            f"batch_size={len(data.inputs)} "
            f"duration={duration:.4f}s"
        )

        return PredictionBatchOutput(
            predictions=results
        )

    except Exception as e:
        duration = time.perf_counter() - start_time

        logger.error(
            f"batch_prediction_failed "
            f"request_id={request_id} "
            f"batch_size={len(data.inputs)} "
            f"duration={duration:.4f}s "
            f"error={e}"
        )

        raise PredictionError()


@router.get("/model-info")
def model_info(request: Request):
    model = getattr(request.app.state, "model", None)

    if model is None:
        raise PredictionError()

    return {
        "model_type": type(model).__name__,
        "model_version": "1.0",
        "training_date": "2026-09-01",
        "features": [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]
    }