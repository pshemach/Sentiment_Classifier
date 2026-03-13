from pydantic import BaseModel


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float