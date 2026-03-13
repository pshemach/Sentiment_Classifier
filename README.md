# Sentiment Classifier API

A FastAPI service that serves a **DistilBERT-based sentiment classification model** trained using HuggingFace Transformers.

The API predicts whether input text expresses **positive or negative sentiment** and returns a **confidence score**.

---

# Project Structure

```
Sentiment_Classifier/

├── app/
│   ├── main.py          # FastAPI application
│   ├── model.py         # Model loading + prediction logic
│   |── schemas.py       # Pydantic request/response models
│   └── utils.py         # helper functions
│
├── data/
│   └── csv/
│       ├── train.csv
│       └── test.csv
│
├── model/               # Saved trained model
│
├── train.py             # Model training script
├── pyproject.toml       # Dependencies (managed with uv)
└── README.md
```

---

# Requirements

- Python **3.11+**
- **uv** package manager

Install uv:

```
pip install uv
```

or

```
curl -Ls https://astral.sh/uv/install.sh | sh
```

---

# Setup Instructions

Clone the repository:

```
git clone https://github.com/pshemach/Sentiment_Classifier.git
cd Sentiment_Classifier
```

Create a virtual environment and install dependencies using **uv**:

```
uv venv
uv sync
```

Activate the environment

Windows:

```
.venv\Scripts\activate
```

Linux / macOS:

```
source .venv/bin/activate
```

---

# Train the Model

Run the training script:

```
python train.py
```

This will:

1. Load training and testing datasets
2. Preprocess text
3. Fine-tune a DistilBERT model
4. Evaluate the model
5. Save the trained model in the `model/` directory

---

# Evaluation Report

Evaluation metrics on the test dataset:

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.9087 |
| Precision | 0.9086 |
| Recall    | 0.9089 |
| F1 Score  | 0.9087 |

These results indicate balanced performance across both classes.

---

# Start the API Server

Run the FastAPI service:

```
uvicorn app.main:app --reload
```

The API will start at:

```
http://127.0.0.1:8000
```

Interactive documentation:

```
http://127.0.0.1:8000/docs
```

---

# API Endpoints

## Health Check

```
GET /health
```

Response:

```
{ "status": "ok" }
```

---

# Predict Sentiment

```
POST /predict
```

Request:

```
{
  "text": "This movie was amazing"
}
```

Response:

```
{
  "sentiment": "positive",
  "confidence": 0.98
}
```

---

# Batch Prediction

```
POST /predict/batch
```

Request:

```
{
  "texts": [
    "This movie was fantastic",
    "Worst movie ever"
  ]
}
```

Response:

```
[
  {
    "text": "This movie was fantastic",
    "sentiment": "positive",
    "confidence": 0.98
  },
  {
    "text": "Worst movie ever",
    "sentiment": "negative",
    "confidence": 0.97
  }
]
```

If an empty list is sent, the API returns:

```
HTTP 400
{
  "detail": "The 'texts' list cannot be empty."
}
```

---

# Model and Dataset Choice

This project uses **DistilBERT**, a smaller and faster version of BERT. It provides strong language understanding while being more efficient for API-based inference.

The model was fine-tuned on the **IMDb movie reviews dataset**, a widely used benchmark for sentiment analysis containing labeled positive and negative reviews.

Using a pretrained transformer allows the system to leverage **transfer learning**, meaning the model already understands language structure and only needs fine-tuning for sentiment prediction.

---

# Approach

The system fine-tunes a **DistilBERT transformer model** for binary sentiment classification using the HuggingFace Trainer API.

Text inputs are tokenized with a maximum length of **256 tokens**, which captures most relevant context while keeping training efficient.

Batch inference is implemented via the `/predict/batch` endpoint to improve throughput compared to sending requests one-by-one.

With more time, improvements would include **experiment tracking, model versioning, and request batching queues for large-scale inference systems**.
