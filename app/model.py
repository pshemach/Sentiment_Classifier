"""
Prediction script for Sentiment Classification
"""
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentPredictor:

    def __init__(self, model_path="model"):

        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        self.model.eval()

    # ------------------------------------------------
    # Text preprocessing (same as training)
    # ------------------------------------------------
    def clean_text(self, text):

        text = text.lower()

        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ------------------------------------------------
    # Predict sentiment
    # ------------------------------------------------
    def predict(self, text):

        text = self.clean_text(text)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        with torch.no_grad():

            outputs = self.model(**inputs)

            logits = outputs.logits

            probabilities = torch.softmax(logits, dim=1)

            predicted_class = torch.argmax(probabilities).item()

            confidence = probabilities[0][predicted_class].item()

        label = "POSITIVE" if predicted_class == 1 else "NEGATIVE"

        return {
            "text":text,
            "sentiment": label,
            "confidence": round(confidence, 2)
        }
        
    # ------------------------------------------------
    # Batch prediction
    # ------------------------------------------------
    def predict_batch(self, texts):
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        inputs = self.tokenizer(
            cleaned_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
            )
        with torch.no_grad():

            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)

        results = []

        for i, text in enumerate(texts):

            prediction = torch.argmax(probs[i]).item()

            confidence = probs[i][prediction].item()

            sentiment = "positive" if prediction == 1 else "negative"

            results.append({
                "text": text,
                "sentiment": sentiment,
                "confidence": round(confidence, 4)
            })

        return results