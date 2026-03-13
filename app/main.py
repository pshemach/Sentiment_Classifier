from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
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

# Enable CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

# ------------------------------------------------
# Startup event (load model once)
# ------------------------------------------------
@app.on_event("startup")
def load_model():
    global model                          # create global model variable for load the model once 
    model = SentimentPredictor("model")  

# ------------------------------------------------
# Health endpoint
# ------------------------------------------------
@app.get("/health")
def health():
    try:
        if model is None:    # check whether the model is loaded correctly 
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# ------------------------------------------------
# Prediction endpoint
# ------------------------------------------------
@app.post("/predict", status_code=status.HTTP_200_OK, response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        result = model.predict(request.text)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
            
# ------------------------------
# Batch prediction
# ------------------------------
@app.post("/predict/batch", status_code=status.HTTP_200_OK, response_model=List[BatchPrediction])
def predict_batch(request: BatchPredictRequest):
    try:
        if len(request.texts) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The 'texts' list cannot be empty.")
        
        results = model.predict_batch(request.texts)

        return results
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))