import pandas as pd
import re
import numpy as np
import logging
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from app.utils import clean_text

# ---------------------------------------------------
# Logger Setup
# ---------------------------------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),  # save to file
        logging.StreamHandler()  # also print to console
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
class Config:

    TRAIN_PATH = "data/csv/train.csv"
    TEST_PATH = "data/csv/test.csv"

    MODEL_NAME = "distilbert-base-uncased"
    OUTPUT_DIR = "model"

    MAX_LENGTH = 256
    TRAIN_BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 2e-5


# ---------------------------------------------------
# Sentiment Trainer Class
# ---------------------------------------------------
class SentimentTrainer:
    def __init__(self, config: Config):
        self.config = config
        
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        logger.info("Loading model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.MODEL_NAME,
            num_labels=2
        )

    # ---------------------------------------------------
    # Load Dataset
    # ---------------------------------------------------
    def load_data(self):

        logger.info("Loading CSV datasets...")

        train_df = pd.read_csv(self.config.TRAIN_PATH)
        test_df = pd.read_csv(self.config.TEST_PATH)

        train_df["text"] = train_df["text"].apply(clean_text)
        test_df["text"] = test_df["text"].apply(clean_text)
        
        train_df["token_length"] = train_df["text"].apply(lambda x: len(self.tokenizer.tokenize(x, truncation=False)))
        train_df = train_df[train_df["token_length"] < 512] # remove outliers

        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)

        return train_dataset, test_dataset

    # ---------------------------------------------------
    # Tokenization
    # ---------------------------------------------------
    def tokenize(self, examples):

        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.config.MAX_LENGTH
        )

    def prepare_dataset(self, train_dataset, test_dataset):

        logger.info("Tokenizing dataset...")

        train_dataset = train_dataset.map(self.tokenize, batched=True)
        test_dataset = test_dataset.map(self.tokenize, batched=True)

        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"]
        )

        test_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"]
        )

        return train_dataset, test_dataset

    # ---------------------------------------------------
    # Evaluation Metrics
    # ---------------------------------------------------
    def compute_metrics(self, eval_pred):

        logits, labels = eval_pred
        
        predictions = np.argmax(logits, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="binary"
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # ---------------------------------------------------
    # Training
    # ---------------------------------------------------
    def train(self):

        train_dataset, test_dataset = self.load_data()

        train_dataset, test_dataset = self.prepare_dataset(
            train_dataset,
            test_dataset
        )

        training_args = TrainingArguments(
            output_dir=self.config.OUTPUT_DIR,
            learning_rate=self.config.LEARNING_RATE,
            per_device_train_batch_size=self.config.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=self.config.TRAIN_BATCH_SIZE,
            num_train_epochs=self.config.EPOCHS,
            weight_decay=0.01,
            save_strategy="epoch"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            processing_class=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        logger.info("Starting training...")

        trainer.train()

        logger.info("Evaluating model...")

        results = trainer.evaluate()

        logger.info(f"Evaluation Results: {results}")

        logger.info("Saving model...")

        trainer.save_model(self.config.OUTPUT_DIR)
        self.tokenizer.save_pretrained(self.config.OUTPUT_DIR)

        logger.info("Training complete!")


def main():

    config = Config()

    trainer = SentimentTrainer(config)

    trainer.train()


if __name__ == "__main__":
    main()