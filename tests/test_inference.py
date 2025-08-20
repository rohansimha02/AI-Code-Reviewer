"""
Tests for inference functionality.
"""

import pytest
import torch
from unittest.mock import Mock, patch
import tempfile
import os


def test_inference_initialization():
    """Test inference class initialization."""
    from serving.inference import CodeReviewerInference
    
    # Test initialization
    inference = CodeReviewerInference("test_model_path")
    
    assert inference.model_path == "test_model_path"
    assert inference.model is None
    assert inference.tokenizer is None
    assert inference.model_loaded is False
    assert inference.max_length == 512


@patch('serving.inference.AutoModelForSequenceClassification.from_pretrained')
@patch('serving.inference.AutoTokenizer.from_pretrained')
def test_model_loading(mock_tokenizer, mock_model):
    """Test model loading functionality."""
    from serving.inference import CodeReviewerInference
    
    # Mock the model and tokenizer
    mock_model_instance = Mock()
    mock_tokenizer_instance = Mock()
    mock_model.return_value = mock_model_instance
    mock_tokenizer.return_value = mock_tokenizer_instance
    
    # Mock device
    mock_model_instance.to.return_value = mock_model_instance
    
    # Mock tokenizer attributes
    mock_tokenizer_instance.pad_token = None
    mock_tokenizer_instance.model_max_length = 512
    
    inference = CodeReviewerInference()
    
    # Mock that model path exists
    with patch('pathlib.Path.exists', return_value=True):
        # Mock torch.cuda.is_available
        with patch('torch.cuda.is_available', return_value=False):
            success = inference.load_model()
    
    assert success is True
    assert inference.model_loaded is True
    assert inference.model is not None
    assert inference.tokenizer is not None


def test_prediction_without_model():
    """Test prediction when model is not loaded."""
    from serving.inference import CodeReviewerInference
    
    inference = CodeReviewerInference()
    
    with pytest.raises(RuntimeError, match="Model not loaded"):
        inference.predict("def test(): pass")


@patch('serving.inference.AutoModelForSequenceClassification.from_pretrained')
@patch('serving.inference.AutoTokenizer.from_pretrained')
def test_prediction_functionality(mock_tokenizer, mock_model):
    """Test prediction functionality."""
    from serving.inference import CodeReviewerInference
    import torch.nn.functional as F
    
    # Mock the model and tokenizer
    mock_model_instance = Mock()
    mock_tokenizer_instance = Mock()
    mock_model.return_value = mock_model_instance
    mock_tokenizer.return_value = mock_tokenizer_instance
    
    # Mock device
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.parameters.return_value = [torch.tensor([1.0])]  # Mock parameters
    
    # Mock tokenizer
    mock_tokenizer_instance.pad_token = None
    mock_tokenizer_instance.model_max_length = 512
    
    # Mock tokenization
    mock_tokenizer_instance.return_value = {
        'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
    }
    
    # Mock model output
    mock_output = Mock()
    mock_output.logits = torch.tensor([[0.1, 0.9]])  # Higher probability for class 1 (buggy)
    mock_model_instance.return_value = mock_output
    
    inference = CodeReviewerInference()
    
    # Mock that model path exists
    with patch('pathlib.Path.exists', return_value=True):
        # Mock torch.cuda.is_available
        with patch('torch.cuda.is_available', return_value=False):
            inference.load_model()
    
    # Test prediction
    with torch.no_grad():
        label, confidence = inference.predict("def test(): return 1 / 0")
    
    assert label in ["buggy", "clean"]
    assert 0.0 <= confidence <= 1.0


def test_prediction_validation():
    """Test prediction input validation."""
    from serving.inference import CodeReviewerInference
    
    inference = CodeReviewerInference()
    
    # Mock model as loaded
    inference.model_loaded = True
    inference.model = Mock()
    inference.tokenizer = Mock()
    inference.device = torch.device('cpu')
    
    # Test empty code
    with pytest.raises(ValueError, match="Code cannot be empty"):
        inference.predict("")
    
    with pytest.raises(ValueError, match="Code cannot be empty"):
        inference.predict("   ")


@patch('serving.inference.AutoModelForSequenceClassification.from_pretrained')
@patch('serving.inference.AutoTokenizer.from_pretrained')
def test_batch_prediction(mock_tokenizer, mock_model):
    """Test batch prediction functionality."""
    from serving.inference import CodeReviewerInference
    
    # Mock the model and tokenizer
    mock_model_instance = Mock()
    mock_tokenizer_instance = Mock()
    mock_model.return_value = mock_model_instance
    mock_tokenizer.return_value = mock_tokenizer_instance
    
    # Mock device
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.parameters.return_value = [torch.tensor([1.0])]
    
    # Mock tokenizer
    mock_tokenizer_instance.pad_token = None
    mock_tokenizer_instance.model_max_length = 512
    
    # Mock tokenization for batch
    mock_tokenizer_instance.return_value = {
        'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6]]),
        'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 1]])
    }
    
    # Mock model output for batch
    mock_output = Mock()
    mock_output.logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])  # First buggy, second clean
    mock_model_instance.return_value = mock_output
    
    inference = CodeReviewerInference()
    
    # Mock that model path exists
    with patch('pathlib.Path.exists', return_value=True):
        with patch('torch.cuda.is_available', return_value=False):
            inference.load_model()
    
    # Test batch prediction
    codes = ["def test1(): return 1 / 0", "def test2(): return 42"]
    
    with torch.no_grad():
        predictions = inference.predict_batch(codes)
    
    assert len(predictions) == 2
    for label, confidence in predictions:
        assert label in ["buggy", "clean"]
        assert 0.0 <= confidence <= 1.0


def test_batch_prediction_validation():
    """Test batch prediction validation."""
    from serving.inference import CodeReviewerInference
    
    inference = CodeReviewerInference()
    inference.model_loaded = True
    inference.model = Mock()
    inference.tokenizer = Mock()
    inference.device = torch.device('cpu')
    
    # Test empty batch
    with pytest.raises(RuntimeError, match="Model not loaded"):
        inference.predict_batch([])


def test_model_info():
    """Test model information retrieval."""
    from serving.inference import CodeReviewerInference
    
    inference = CodeReviewerInference()
    
    # Test when model not loaded
    info = inference.get_model_info()
    assert "error" in info
    
    # Test when model loaded
    inference.model_loaded = True
    inference.device = torch.device('cpu')
    inference.max_length = 512
    
    info = inference.get_model_info()
    assert info["model_name"] == "AI Code Reviewer"
    assert info["base_model"] == "microsoft/codebert-base"
    assert info["task"] == "sequence-classification"
    assert info["max_length"] == 512


def test_global_inference_instance():
    """Test global inference instance management."""
    from serving.inference import get_inference_instance
    
    # Get instance
    instance1 = get_inference_instance()
    instance2 = get_inference_instance()
    
    # Should be the same instance
    assert instance1 is instance2


@patch('serving.inference.CodeReviewerInference')
def test_predict_code_function(mock_inference_class):
    """Test the predict_code convenience function."""
    from serving.inference import predict_code
    
    # Mock inference instance
    mock_instance = Mock()
    mock_instance.model_loaded = True
    mock_instance.predict.return_value = ("buggy", 0.95)
    mock_inference_class.return_value = mock_instance
    
    # Test prediction
    label, confidence = predict_code("def test(): return 1 / 0")
    
    assert label == "buggy"
    assert confidence == 0.95
    mock_instance.predict.assert_called_once_with("def test(): return 1 / 0")


if __name__ == "__main__":
    pytest.main([__file__])
