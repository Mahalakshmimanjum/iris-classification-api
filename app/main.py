from contextlib import asynccontextmanager
import time
import uuid

import joblib
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.exceptions import PredictionError
from app.logging_config import setup_logging
from app.routers.v1 import router as v1_router
from app.routers.v2 import router as v2_router


logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML model...")
    logger.info(f"Model path: {settings.MODEL_PATH}")

    app.state.model = joblib.load(
        settings.MODEL_PATH
    )

    logger.info("ML model loaded successfully!")

    yield


app = FastAPI(
    title=settings.API_TITLE,
    lifespan=lifespan
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start_time

    logger.info(
        f"request_id={request_id} "
        f"method={request.method} "
        f"path={request.url.path} "
        f"status_code={response.status_code} "
        f"duration={duration:.4f}s"
    )

    return response


@app.exception_handler(PredictionError)
async def prediction_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Prediction failed"}
    )


@app.get("/")
def root():
    return {"message": "ML API is alive"}


app.include_router(v1_router)
app.include_router(v2_router)