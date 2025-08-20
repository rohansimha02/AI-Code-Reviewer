"""
Tests for tokenizer functionality.
"""

import pytest
import torch
from transformers import AutoTokenizer


def test_tokenizer_initialization():
    """Test tokenizer initialization."""
    # Test with CodeBERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    assert tokenizer is not None
    assert hasattr(tokenizer, 'vocab_size')
    assert hasattr(tokenizer, 'model_max_length')
    
    # Check that padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    assert tokenizer.pad_token is not None


def test_tokenizer_truncation():
    """Test tokenizer truncation."""
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test with short code
    short_code = "def test(): pass"
    tokens = tokenizer(
        short_code,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    assert tokens['input_ids'].shape[1] <= 512
    assert tokens['attention_mask'].shape[1] <= 512
    
    # Test with long code (should be truncated)
    long_code = "def test():\n" + "    pass\n" * 1000
    tokens = tokenizer(
        long_code,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    assert tokens['input_ids'].shape[1] <= 512
    assert tokens['attention_mask'].shape[1] <= 512


def test_tokenizer_padding():
    """Test tokenizer padding."""
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test batch tokenization with padding
    codes = [
        "def short(): pass",
        "def longer_function(x, y):\n    return x + y",
        "def very_long_function():\n    result = 0\n    for i in range(100):\n        result += i\n    return result"
    ]
    
    tokens = tokenizer(
        codes,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # All sequences should have the same length
    assert tokens['input_ids'].shape[0] == len(codes)
    assert tokens['attention_mask'].shape[0] == len(codes)
    
    # Check that padding is applied
    max_len = tokens['input_ids'].shape[1]
    assert max_len <= 512


def test_tokenizer_tensor_shapes():
    """Test that tokenizer returns correct tensor shapes."""
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Single input
    code = "def test(): return 42"
    tokens = tokenizer(
        code,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    assert isinstance(tokens['input_ids'], torch.Tensor)
    assert isinstance(tokens['attention_mask'], torch.Tensor)
    assert tokens['input_ids'].dim() == 2  # [batch_size, sequence_length]
    assert tokens['attention_mask'].dim() == 2
    
    # Batch input
    codes = ["def test1(): pass", "def test2(): return 0"]
    tokens = tokenizer(
        codes,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    assert tokens['input_ids'].dim() == 2
    assert tokens['attention_mask'].dim() == 2
    assert tokens['input_ids'].shape[0] == len(codes)


def test_tokenizer_decode():
    """Test tokenizer decode functionality."""
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Original code
    original_code = "def test(): return 42"
    
    # Tokenize
    tokens = tokenizer(
        original_code,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Decode
    decoded = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
    
    # Should be similar to original (may have slight differences due to tokenization)
    assert "def" in decoded
    assert "test" in decoded


def test_tokenizer_special_tokens():
    """Test that special tokens are handled correctly."""
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check that special tokens are defined
    assert tokenizer.pad_token is not None
    assert tokenizer.eos_token is not None
    assert tokenizer.bos_token is not None
    
    # Test that padding token is in vocabulary
    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    assert pad_token_id != tokenizer.unk_token_id


def test_tokenizer_edge_cases():
    """Test tokenizer with edge cases."""
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Empty string
    tokens = tokenizer(
        "",
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    assert tokens['input_ids'].shape[1] <= 512
    
    # Very long string
    very_long_code = "def test():\n" + "    pass\n" * 2000
    tokens = tokenizer(
        very_long_code,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    assert tokens['input_ids'].shape[1] <= 512
    
    # Special characters
    special_code = "def test():\n    # Comment with special chars: @#$%^&*()\n    return 'string with \"quotes\"'"
    tokens = tokenizer(
        special_code,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    assert tokens['input_ids'].shape[1] <= 512


if __name__ == "__main__":
    pytest.main([__file__])
