import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple, List
import logging

from app.models.model import HoaxDetectionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Args:
        data_path: Path to the CSV dataset
        
    Returns:
        DataFrame containing the dataset
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    # Determine file type and load accordingly
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
        df = pd.read_excel(data_path)
    else:
        raise ValueError("Unsupported file format. Please provide CSV or Excel file.")
    
    logger.info(f"Loaded dataset with {len(df)} samples")
    return df

def prepare_data(
    df: pd.DataFrame, 
    text_column: str = 'text', 
    label_column: str = 'label',
    test_size: float = 0.2,
    val_size: float = 0.1
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Prepare data for model training and evaluation.
    
    Args:
        df: DataFrame containing the dataset
        text_column: Name of the column containing text data
        label_column: Name of the column containing labels
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        
    Returns:
        Tuple containing train, validation, and test data and labels
    """
    # Check if columns exist
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in dataset")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset")
    
    # Extract texts and labels
    texts = df[text_column].tolist()
    
    # Convert labels to integers if they are not already
    if isinstance(df[label_column].iloc[0], str):
        # Assuming binary classification with labels like "hoax"/"not_hoax" or "1"/"0"
        label_map = {"hoax": 1, "not_hoax": 0, "1": 1, "0": 0, "true": 1, "false": 0}
        labels = [label_map.get(str(label).lower(), int(label)) for label in df[label_column]]
    else:
        labels = df[label_column].tolist()
    
    # Split data into train+val and test
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Split train+val into train and val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, 
        test_size=val_size/(1-test_size),  # Adjust val_size relative to train+val size
        random_state=42, 
        stratify=train_val_labels
    )
    
    logger.info(f"Data split: {len(train_texts)} train, {len(val_texts)} validation, {len(test_texts)} test samples")
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

def train_and_evaluate(
    data_path: str,
    model_path: str = None,
    pretrained_model: str = "indolem/indobert-base-uncased",
    text_column: str = 'text',
    label_column: str = 'label',
    epochs: int = 3,
    batch_size: int = 8
) -> Dict[str, Any]:
    """
    Train and evaluate the hoax detection model.
    
    Args:
        data_path: Path to the dataset
        model_path: Path to save the model
        pretrained_model: Pretrained model to use
        text_column: Name of the column containing text data
        label_column: Name of the column containing labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Dictionary with training and evaluation results
    """
    # Load data
    df = load_data(data_path)
    
    # Prepare data
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = prepare_data(
        df, text_column, label_column
    )
    
    # Initialize model
    model = HoaxDetectionModel(
        model_path=model_path if model_path else "models/hoax_detection_model",
        pretrained_model=pretrained_model
    )
    
    # Fine-tune model
    logger.info(f"Fine-tuning model with {len(train_texts)} samples...")
    train_results = model.fine_tune(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Evaluate model
    logger.info(f"Evaluating model with {len(test_texts)} samples...")
    eval_results = model.evaluate(test_texts, test_labels)
    
    # Save model
    model.save_model()
    
    return {
        "train_results": train_results,
        "eval_results": eval_results,
        "model_path": model.model_path
    }
