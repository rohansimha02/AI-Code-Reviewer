"""
Tokenizer utilities for AI Code Reviewer.
"""

import logging
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class CodeTokenizer:
    """Tokenizer wrapper for code processing."""
    
    def __init__(self, model_name: str, max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Initialized tokenizer for {model_name}")
        logger.info(f"Max length: {max_length}")
    
    def tokenize(self, texts: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        """Tokenize code texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize with truncation and padding
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
            **kwargs
        )
        
        return tokenized
    
    def tokenize_batch(self, texts: List[str], **kwargs) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts."""
        return self.tokenize(texts, **kwargs)
    
    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """Decode token IDs back to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.vocab_size
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer to directory."""
        self.tokenizer.save_pretrained(save_directory)
        logger.info(f"Tokenizer saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, save_directory: str, max_length: int = 512):
        """Load tokenizer from directory."""
        tokenizer = AutoTokenizer.from_pretrained(save_directory)
        instance = cls.__new__(cls)
        instance.model_name = save_directory
        instance.max_length = max_length
        instance.tokenizer = tokenizer
        return instance


def create_tokenizer(config: Dict) -> CodeTokenizer:
    """Create tokenizer from configuration."""
    model_name = config.get("base_model", "microsoft/codebert-base")
    max_length = config.get("max_length", 512)
    
    return CodeTokenizer(model_name, max_length)


def analyze_tokenization(tokenizer: CodeTokenizer, sample_texts: List[str]):
    """Analyze tokenization statistics for sample texts."""
    logger.info("Tokenization Analysis:")
    logger.info("=" * 40)
    
    total_tokens = 0
    total_texts = len(sample_texts)
    
    for i, text in enumerate(sample_texts[:5]):  # Analyze first 5 samples
        tokenized = tokenizer.tokenize(text)
        num_tokens = tokenized['input_ids'].shape[1]
        total_tokens += num_tokens
        
        logger.info(f"Sample {i+1}: {num_tokens} tokens")
        if num_tokens > tokenizer.max_length:
            logger.warning(f"Sample {i+1} exceeds max length ({tokenizer.max_length})")
    
    avg_tokens = total_tokens / min(5, total_texts)
    logger.info(f"Average tokens per sample: {avg_tokens:.1f}")
    logger.info(f"Max length setting: {tokenizer.max_length}")
    
    if avg_tokens > tokenizer.max_length * 0.8:
        logger.warning("Many samples are close to max length - consider increasing max_length")
    
    logger.info("=" * 40)
