"""
Inference module for AI Code Reviewer.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class CodeReviewerInference:
    """Inference class for AI Code Reviewer."""
    
    def __init__(self, model_path: str = "serving/model_store/best_model"):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.device = None
        self.max_length = 512
        self.model_loaded = False
        
        logger.info(f"Initializing inference with model path: {model_path}")
    
    def load_model(self) -> bool:
        """Load the trained model and tokenizer."""
        try:
            if not self.model_path.exists():
                logger.error(f"Model path does not exist: {self.model_path}")
                return False
            
            # Determine device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
            
            logger.info(f"Using device: {self.device}")
            
            # Load model and tokenizer
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Set max length from tokenizer if available
            if hasattr(self.tokenizer, 'model_max_length'):
                self.max_length = min(self.tokenizer.model_max_length, 512)
            
            self.model_loaded = True
            logger.info("Model loaded successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model_loaded = False
            return False
    
    def predict(self, code: str) -> Tuple[str, float]:
        """Predict whether code is buggy or clean."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not code.strip():
            raise ValueError("Code cannot be empty")
        
        # Tokenize input
        inputs = self.tokenizer(
            code,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = torch.max(probabilities, dim=-1).values.item()
        
        # Map to labels
        label = "buggy" if predicted_class == 1 else "clean"
        
        return label, confidence
    
    def predict_batch(self, codes: list) -> list:
        """Predict for a batch of code samples."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not codes:
            return []
        
        # Tokenize inputs
        inputs = self.tokenizer(
            codes,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            
            predicted_classes = torch.argmax(probabilities, dim=-1)
            confidences = torch.max(probabilities, dim=-1).values
        
        # Convert to list of predictions
        predictions = []
        for pred_class, confidence in zip(predicted_classes, confidences):
            label = "buggy" if pred_class.item() == 1 else "clean"
            predictions.append((label, confidence.item()))
        
        return predictions
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self.model_loaded:
            return {"error": "Model not loaded"}
        
        return {
            "model_name": "AI Code Reviewer",
            "base_model": "microsoft/codebert-base",
            "task": "sequence-classification",
            "max_length": self.max_length,
            "device": str(self.device),
            "model_path": str(self.model_path)
        }


# Global inference instance
_inference_instance: Optional[CodeReviewerInference] = None


def get_inference_instance() -> CodeReviewerInference:
    """Get or create the global inference instance."""
    global _inference_instance
    
    if _inference_instance is None:
        _inference_instance = CodeReviewerInference()
    
    return _inference_instance


def predict_code(code: str) -> Tuple[str, float]:
    """Predict whether code is buggy or clean."""
    inference = get_inference_instance()
    
    if not inference.model_loaded:
        if not inference.load_model():
            raise RuntimeError("Failed to load model")
    
    return inference.predict(code)


def load_model_for_inference(model_path: str = "serving/model_store/best_model") -> bool:
    """Load model for inference."""
    inference = get_inference_instance()
    return inference.load_model()
