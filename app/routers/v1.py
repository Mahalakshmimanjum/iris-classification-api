from fastapi import APIRouter, Request

from app.logging_config import setup_logging
from app.models.schemas import PredictionInput, PredictionOutput
from app.exceptions import PredictionError




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