"""
FastAPI application for AI Code Reviewer.
"""

import logging
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import CodeRequest, CodeResponse, HealthResponse, ModelInfoResponse
from .inference import get_inference_instance, load_model_for_inference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Code Reviewer API",
    description="API for detecting buggy vs clean Python code using CodeBERT",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    logger.info("Starting AI Code Reviewer API...")
    
    # Try to load the model
    inference = get_inference_instance()
    if inference.load_model():
        logger.info("Model loaded successfully on startup")
    else:
        logger.warning("Model not loaded on startup. It will be loaded on first request.")


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Code Reviewer API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    inference = get_inference_instance()
    
    return HealthResponse(
        status="ok",
        model_loaded=inference.model_loaded
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information."""
    inference = get_inference_instance()
    
    if not inference.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = inference.get_model_info()
    
    return ModelInfoResponse(
        model_name=info["model_name"],
        base_model=info["base_model"],
        task=info["task"],
        max_length=info["max_length"]
    )


@app.post("/predict", response_model=CodeResponse)
async def predict_code(request: CodeRequest):
    """Predict whether code is buggy or clean."""
    try:
        inference = get_inference_instance()
        
        # Load model if not already loaded
        if not inference.model_loaded:
            if not inference.load_model():
                raise HTTPException(
                    status_code=503, 
                    detail="Failed to load model. Please ensure the model is trained and available."
                )
        
        # Make prediction
        label, score = inference.predict(request.code)
        
        return CodeResponse(label=label, score=score)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/predict-batch")
async def predict_batch(codes: list[str]):
    """Predict for a batch of code samples."""
    try:
        inference = get_inference_instance()
        
        # Load model if not already loaded
        if not inference.model_loaded:
            if not inference.load_model():
                raise HTTPException(
                    status_code=503, 
                    detail="Failed to load model. Please ensure the model is trained and available."
                )
        
        # Validate input
        if not codes:
            raise HTTPException(status_code=400, detail="Empty batch")
        
        if len(codes) > 100:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
        
        # Make predictions
        predictions = inference.predict_batch(codes)
        
        # Format response
        results = []
        for i, (label, score) in enumerate(predictions):
            results.append({
                "index": i,
                "code": codes[i][:100] + "..." if len(codes[i]) > 100 else codes[i],  # Truncate for response
                "label": label,
                "score": score
            })
        
        return {"predictions": results}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/examples")
async def get_examples():
    """Get example code snippets for testing."""
    return {
        "examples": [
            {
                "name": "Buggy Division by Zero",
                "code": "def buggy_function():\n    return 1 / 0",
                "expected": "buggy"
            },
            {
                "name": "Clean Function",
                "code": "def clean_function(x, y):\n    return x + y",
                "expected": "clean"
            },
            {
                "name": "Buggy List Access",
                "code": "def buggy_list_access(lst):\n    return lst[10]",
                "expected": "buggy"
            },
            {
                "name": "Clean List Access",
                "code": "def clean_list_access(lst, index):\n    if 0 <= index < len(lst):\n        return lst[index]\n    return None",
                "expected": "clean"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
