"""
Prediction script for Sentiment Classification
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .utils import clean_text

# ------------------------------------------------
# Sentiment prediction class
# ------------------------------------------------
class SentimentPredictor:
    """Sentiment prediction class using a locally saved DistilBERT model."""
    def __init__(self, model_path="model"):
        try:
            self.model_path = model_path
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model or tokenizer from {model_path}") from e

    def predict(self, text):
        """
        Predict sentiment for a single text input.

        Args:
            text (str): Input text to classify

        Returns:
            dict: Prediction result containing:
                - "text": input text
                - "sentiment": "positive" or "negative"
                - "confidence": probability of predicted class (0.00–1.00)
        """
        cleaned_text = clean_text(text)

        inputs = self.tokenizer(
            cleaned_text,
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

        label = "positive" if predicted_class == 1 else "negative"

        return {
            "text":text,
            "sentiment": label,
            "confidence": round(confidence, 2)
        }
        
    def predict_batch(self, texts):
        """
        Predict sentiment for batch inference.

        Args:
            texts (list[str]): List of input texts to classify

        Returns:
            list[dict]: List of prediction results, each containing:
                - "text": original input text
                - "sentiment": "positive" or "negative"
                - "confidence": probability of predicted class (0.00–1.00)
        """
        cleaned_texts = [clean_text(text) for text in texts]
        
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
                "confidence": round(confidence, 2)
            })

        return results