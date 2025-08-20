"""
CodeBERT classifier model for AI Code Reviewer.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class CodeBERTClassifier:
    """CodeBERT-based classifier for bug detection."""
    
    def __init__(self, model_name: str = "microsoft/codebert-base", num_labels: int = 2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initializing CodeBERT classifier with {model_name}")
        logger.info(f"Number of labels: {num_labels}")
    
    def load_pretrained(self, model_path: Optional[str] = None):
        """Load pretrained model and tokenizer."""
        if model_path:
            # Load from local path
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            # Load from HuggingFace hub
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=self.num_labels
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info(f"Loaded model from HuggingFace hub: {self.model_name}")
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def predict(self, code: str, max_length: int = 512) -> Tuple[str, float]:
        """Predict whether code is buggy or clean."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_pretrained() first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            code,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
        
        # Get prediction and confidence
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = torch.max(probabilities, dim=-1).values.item()
        
        # Map to labels
        label = "buggy" if predicted_class == 1 else "clean"
        
        return label, confidence
    
    def predict_batch(self, codes: list, max_length: int = 512) -> list:
        """Predict for a batch of code samples."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_pretrained() first.")
        
        # Tokenize inputs
        inputs = self.tokenizer(
            codes,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
        
        # Get predictions and confidences
        predicted_classes = torch.argmax(probabilities, dim=-1)
        confidences = torch.max(probabilities, dim=-1).values
        
        # Convert to list of predictions
        predictions = []
        for pred_class, confidence in zip(predicted_classes, confidences):
            label = "buggy" if pred_class.item() == 1 else "clean"
            predictions.append((label, confidence.item()))
        
        return predictions
    
    def save_pretrained(self, save_directory: str):
        """Save model and tokenizer to directory."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_pretrained() first.")
        
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        logger.info(f"Model and tokenizer saved to {save_directory}")
    
    def to_device(self, device: str):
        """Move model to specified device."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_pretrained() first.")
        
        self.model = self.model.to(device)
        logger.info(f"Model moved to device: {device}")


def create_model(config: Dict) -> CodeBERTClassifier:
    """Create model from configuration."""
    model_name = config.get("base_model", "microsoft/codebert-base")
    num_labels = config.get("num_labels", 2)
    
    return CodeBERTClassifier(model_name, num_labels)


def load_model_for_inference(model_path: str) -> CodeBERTClassifier:
    """Load trained model for inference."""
    classifier = CodeBERTClassifier()
    classifier.load_pretrained(model_path)
    return classifier


def compute_logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits to probabilities using softmax."""
    return F.softmax(logits, dim=-1)


def get_prediction_from_logits(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get predictions and probabilities from logits."""
    probabilities = compute_logits_to_probabilities(logits)
    predictions = torch.argmax(probabilities, dim=-1)
    return predictions, probabilities
