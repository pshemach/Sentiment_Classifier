from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    
class BatchPredictRequest(BaseModel):
    texts: List[str]


class BatchPrediction(BaseModel):
    text: str
    sentiment: str
    confidence: float