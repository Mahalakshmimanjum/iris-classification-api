from typing import List

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)


# v1 response schema
class PredictionOutput(BaseModel):
    prediction: int
    confidence: float
    model_version: str
    request_id: str


class PredictionBatchInput(BaseModel):
    inputs: List[PredictionInput]


class PredictionBatchOutput(BaseModel):
    predictions: List[PredictionOutput]


# v2 response schema
class PredictionV2Output(BaseModel):
    prediction: int
    probabilities: dict[str, float]
    model_version: str
    request_id: str