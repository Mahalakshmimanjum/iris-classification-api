from typing import List

from pydantic import BaseModel, ConfigDict, Field


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)


# v1 response schema
class PredictionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: int
    confidence: float
    model_version: str
    request_id: str


class PredictionBatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: List[PredictionInput]


class PredictionBatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predictions: List[PredictionOutput]


# v2 response schema
class PredictionV2Output(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: int
    probabilities: dict[str, float]
    model_version: str
    request_id: str