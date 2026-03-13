from fastapi import FastAPI
from app.schemas import PredictRequest, PredictResponse
from app.model import SentimentPredictor


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
    
    return PredictResponse(
        text = result['text'],
        sentiment=result['sentiment'],
        confidence=result['confidence']
    )