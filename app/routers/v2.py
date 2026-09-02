import time

from fastapi import APIRouter, Request

from app.exceptions import PredictionError
from app.logging_config import setup_logging
from app.models.schemas import (
    PredictionInput,
    PredictionV2Output,
)

logger = setup_logging()

router = APIRouter(prefix="/api/v2")


@router.post(
    "/predict",
    response_model=PredictionV2Output
)
def predict_v2(
    data: PredictionInput,
    request: Request
):
    request_id = request.state.request_id
    start_time = time.perf_counter()

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

        probability_distribution = {
            str(index): float(probability)
            for index, probability in enumerate(probabilities[0])
        }

        duration = time.perf_counter() - start_time

        logger.info(
            f"v2_prediction_success "
            f"request_id={request_id} "
            f"prediction={int(prediction[0])} "
            f"duration={duration:.4f}s"
        )

        return PredictionV2Output(
            prediction=int(prediction[0]),
            probabilities=probability_distribution,
            model_version="2.0",
            request_id=request_id
        )

    except Exception as e:
        logger.error(
            f"v2_prediction_failed "
            f"request_id={request_id} "
            f"error={e}"
        )

        raise PredictionError()