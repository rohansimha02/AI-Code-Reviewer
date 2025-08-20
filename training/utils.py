"""
Training utilities for AI Code Reviewer.
"""

import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set deterministic behavior for CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'precision_clean': precision_per_class[0],
        'recall_clean': recall_per_class[0],
        'f1_clean': f1_per_class[0],
        'precision_buggy': precision_per_class[1],
        'recall_buggy': recall_per_class[1],
        'f1_buggy': f1_per_class[1],
    }


def get_device(device_config: str = "auto") -> torch.device:
    """Get the appropriate device for training."""
    if device_config == "cpu":
        return torch.device("cpu")
    elif device_config == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_config == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif device_config == "auto":
        # Prefer MPS on Apple Silicon, fallback to CPU
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def calculate_class_weights(labels: List[int]) -> torch.Tensor:
    """Calculate class weights for imbalanced datasets."""
    from collections import Counter
    
    label_counts = Counter(labels)
    total_samples = len(labels)
    
    # Calculate weights inversely proportional to class frequencies
    weights = []
    for label in sorted(label_counts.keys()):
        weight = total_samples / (len(label_counts) * label_counts[label])
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


class EarlyStoppingCallback(TrainerCallback):
    """Custom callback for early stopping based on validation metrics."""
    
    def __init__(self, patience: int = 3, threshold: float = 0.001):
        self.patience = patience
        self.threshold = threshold
        self.best_metric = None
        self.wait = 0
        self.stopped_epoch = 0
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        
        eval_metric = metrics.get("eval_f1_macro", 0.0)
        
        if self.best_metric is None:
            self.best_metric = eval_metric
        elif eval_metric > self.best_metric + self.threshold:
            self.best_metric = eval_metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                control.should_training_stop = True
                self.stopped_epoch = state.epoch
                logger.info(f"Early stopping triggered at epoch {state.epoch}")


def save_model_card(model_dir: str, config: Dict, metrics: Optional[Dict] = None):
    """Save model card with metadata."""
    import json
    from datetime import datetime
    
    model_card = {
        "name": config.get("model_card", {}).get("name", "AI Code Reviewer"),
        "description": config.get("model_card", {}).get("description", ""),
        "version": config.get("model_card", {}).get("version", "1.0.0"),
        "author": config.get("model_card", {}).get("author", "AI Code Reviewer Team"),
        "license": config.get("model_card", {}).get("license", "MIT"),
        "dataset": config.get("model_card", {}).get("dataset", "BugsInPy + QuixBugs"),
        "base_model": config.get("base_model", "microsoft/codebert-base"),
        "task": config.get("model_card", {}).get("task", "sequence-classification"),
        "language": config.get("model_card", {}).get("language", "python"),
        "training_config": {
            "batch_size": config.get("batch_size", 8),
            "learning_rate": config.get("learning_rate", 2e-5),
            "epochs": config.get("epochs", 3),
            "max_length": config.get("max_length", 512),
        },
        "created_at": datetime.now().isoformat(),
    }
    
    if metrics:
        model_card["metrics"] = metrics
    
    model_card_path = os.path.join(model_dir, "model-card.json")
    with open(model_card_path, 'w') as f:
        json.dump(model_card, f, indent=2)
    
    logger.info(f"Model card saved to {model_card_path}")


def log_training_info(config: Dict):
    """Log training configuration information."""
    logger.info("=" * 50)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 50)
    logger.info(f"Base model: {config.get('base_model')}")
    logger.info(f"Max length: {config.get('max_length')}")
    logger.info(f"Batch size: {config.get('batch_size')}")
    logger.info(f"Learning rate: {config.get('learning_rate')}")
    logger.info(f"Epochs: {config.get('epochs')}")
    logger.info(f"Device: {config.get('device')}")
    logger.info(f"Seed: {config.get('seed')}")
    logger.info("=" * 50)
