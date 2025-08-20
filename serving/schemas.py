"""
FastAPI schemas for AI Code Reviewer.
"""

from pydantic import BaseModel, Field
from typing import Optional


class CodeRequest(BaseModel):
    """Request schema for code prediction."""
    code: str = Field(..., description="Python code to analyze", min_length=1, max_length=10000)
    
    class Config:
        schema_extra = {
            "example": {
                "code": "def buggy_function():\n    return 1 / 0"
            }
        }


class CodeResponse(BaseModel):
    """Response schema for code prediction."""
    label: str = Field(..., description="Prediction label: 'buggy' or 'clean'")
    score: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    
    class Config:
        schema_extra = {
            "example": {
                "label": "buggy",
                "score": 0.95
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "ok",
                "model_loaded": True
            }
        }


class ModelInfoResponse(BaseModel):
    """Model information response schema."""
    model_name: str = Field(..., description="Name of the model")
    base_model: str = Field(..., description="Base model used")
    task: str = Field(..., description="Task type")
    max_length: int = Field(..., description="Maximum input length")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "AI Code Reviewer",
                "base_model": "microsoft/codebert-base",
                "task": "sequence-classification",
                "max_length": 512
            }
        }
