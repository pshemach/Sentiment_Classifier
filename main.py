from app.model import SentimentPredictor
predictor = SentimentPredictor()

def main():
    print("Hello from sentiment-classifier!")
    
    while True:
        text = input("Enter text (or 'exit'): ")
        
        if text.lower() == "exit":
            break

        result = predictor.predict(text)

        print("\nPrediction:", result, "\n")


if __name__ == "__main__":
    main()