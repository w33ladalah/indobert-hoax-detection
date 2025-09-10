import os
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, Any, List, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants
MODEL_PATH = os.environ.get("MODEL_PATH", "models/hoax_detection_model")
PRETRAINED_MODEL = os.environ.get("PRETRAINED_MODEL", "indolem/indobert-base-uncased")
MAX_LENGTH = 512

class HoaxDetectionModel:
    """
    Class for loading, fine-tuning, evaluating and using the hoax detection model.
    Supports both IndoBERT and XLM-R models.
    """

    def __init__(self, model_path: str = MODEL_PATH, pretrained_model: str = PRETRAINED_MODEL):
        """
        Initialize the hoax detection model.

        Args:
            model_path: Path to save/load the fine-tuned model
            pretrained_model: Pretrained model to use (IndoBERT or XLM-R)
        """
        self.model_path = model_path
        self.pretrained_model = pretrained_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        try:
            # Try to load fine-tuned model
            self.load_model()
            logger.info(f"Loaded fine-tuned model from {model_path}")
        except Exception as e:
            # If fine-tuned model doesn't exist, load pretrained model
            logger.info(f"Fine-tuned model not found. Loading pretrained model {pretrained_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model,
                num_labels=2  # Binary classification: hoax or not hoax
            )
            self.model.to(self.device)

    def load_model(self):
        """Load the fine-tuned model and tokenizer from disk."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)

    def save_model(self):
        """Save the fine-tuned model and tokenizer to disk."""
        os.makedirs(self.model_path, exist_ok=True)
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info(f"Model saved to {self.model_path}")

    def fine_tune(self, train_texts: List[str], train_labels: List[int],
                 val_texts: List[str] = None, val_labels: List[int] = None,
                 epochs: int = 3, batch_size: int = 8):
        """
        Fine-tune the model on hoax detection data.

        Args:
            train_texts: List of training text samples
            train_labels: List of training labels (0: not hoax, 1: hoax)
            val_texts: List of validation text samples
            val_labels: List of validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Prepare datasets
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        train_dataset = HoaxDataset(train_encodings, train_labels)

        # Prepare validation dataset if provided
        val_dataset = None
        if val_texts and val_labels:
            val_encodings = self.tokenizer(
                val_texts,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )
            val_dataset = HoaxDataset(val_encodings, val_labels)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.model_path}_results",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            logging_dir=f"{self.model_path}_logs",
            logging_steps=10,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
        )

        # Define trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )

        # Fine-tune the model
        logger.info("Starting model fine-tuning...")
        trainer.train()

        # Save the fine-tuned model
        self.save_model()

        # Evaluate if validation data is provided
        if val_dataset:
            eval_results = trainer.evaluate()
            logger.info(f"Evaluation results: {eval_results}")
            return eval_results

        return {"status": "Fine-tuning completed successfully"}

    def evaluate(self, test_texts: List[str], test_labels: List[int]) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            test_texts: List of test text samples
            test_labels: List of test labels

        Returns:
            Dictionary with evaluation metrics
        """
        # Encode test data
        test_encodings = self.tokenizer(
            test_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        test_dataset = HoaxDataset(test_encodings, test_labels)

        # Create trainer for evaluation
        trainer = Trainer(
            model=self.model,
            compute_metrics=self.compute_metrics
        )

        # Evaluate
        results = trainer.evaluate(test_dataset)
        return results

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Make a prediction for a single text input.

        Args:
            text: Input text to classify

        Returns:
            Dictionary with prediction results
        """
        # Prepare input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(self.device)

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Get prediction
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

        # Map class to label
        label = "hoax" if predicted_class == 1 else "not_hoax"

        return {
            "label": label,
            "confidence": confidence,
            "class_index": predicted_class
        }

    @staticmethod
    def compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute evaluation metrics from predictions.

        Args:
            pred: Prediction object from Trainer

        Returns:
            Dictionary with metrics
        """
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary'
        )
        acc = accuracy_score(labels, preds)

        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }


class HoaxDataset(torch.utils.data.Dataset):
    """Dataset for hoax detection fine-tuning and evaluation."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
