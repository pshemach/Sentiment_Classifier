from fastapi import FastAPI, HTTPException
from app.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPrediction
)
from app.model import SentimentPredictor
from typing import List


app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0.0"
)

model = None

# ------------------------------------------------
# Startup event (load model once)
# ------------------------------------------------
@app.on_event("startup")
def load_model():
    global model
    model = SentimentPredictor("model")

# ------------------------------------------------
# Health endpoint
# ------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ------------------------------------------------
# Prediction endpoint
# ------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):

    result = model.predict(request.text)
    
    return result
    
# ------------------------------
# Batch prediction
# ------------------------------
@app.post("/predict/batch", response_model=List[BatchPrediction])
def predict_batch(request: BatchPredictRequest):

    if len(request.texts) == 0:
        raise HTTPException(
            status_code=400,
            detail="The 'texts' list cannot be empty."
        )
    results = model.predict_batch(request.texts)

    return results