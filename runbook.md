# AI Code Reviewer - Runbook

This runbook provides step-by-step instructions for setting up, training, and deploying the AI Code Reviewer system.

## Prerequisites

- Python 3.10+
- Git
- Docker (optional, for containerized deployment)
- At least 8GB RAM (16GB recommended for training)
- GPU support (optional, for faster training)

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd ai-code-reviewer

# Setup virtual environment and install dependencies
make setup
```

### 2. Download Datasets

```bash
# Download BugsInPy and QuixBugs datasets
./scripts/99_download_samples.sh
```

### 3. Prepare Data

```bash
# Extract and process all datasets
make data
```

This will:
- Extract buggy and fixed code pairs from BugsInPy
- Extract buggy and fixed code pairs from QuixBugs
- Create project-level splits (train/val/test)
- Generate statistics

### 4. Train Model

```bash
# Train the CodeBERT classifier
make train
```

This will:
- Load the pretrained CodeBERT model
- Fine-tune on the prepared dataset
- Save the best model to `serving/model_store/best_model/`

### 5. Evaluate Model

```bash
# Evaluate on test set and QuixBugs
make eval
```

This will:
- Evaluate on the test set
- Show per-project F1 scores
- Evaluate on QuixBugs for sanity checks

### 6. Serve API

```bash
# Start the FastAPI server
make serve
```

The API will be available at `http://localhost:8000`

### 7. Launch Demo

```bash
# Start the Streamlit demo
make demo
```

The demo will be available at `http://localhost:8501`

## Detailed Instructions

### Data Preparation

#### Manual Dataset Download

If the automatic download fails, manually download the datasets:

```bash
# Create data directories
mkdir -p data/raw

# Download BugsInPy
git clone https://github.com/soarsmu/BugsInPy data/raw/bugsinpy

# Download QuixBugs
git clone https://github.com/jkoppel/QuixBugs data/raw/quixbugs
```

#### Data Extraction

The data extraction scripts process the raw datasets:

```bash
# Extract BugsInPy data
python scripts/01_extract_bugsinpy.py \
    --raw_dir data/raw/bugsinpy \
    --out data/interim/bugsinpy_pairs.parquet \
    --max_chars 4000 \
    --context_lines 15

# Extract QuixBugs data
python scripts/02_extract_quixbugs.py \
    --raw_dir data/raw/quixbugs \
    --out data/interim/quixbugs_pairs.parquet \
    --max_chars 4000
```

#### Data Splitting

Create project-level splits to prevent data leakage:

```bash
python scripts/03_merge_and_split.py \
    --bugsinpy_path data/interim/bugsinpy_pairs.parquet \
    --quixbugs_path data/interim/quixbugs_pairs.parquet \
    --output_dir data/processed \
    --seed 42
```

### Training Configuration

The training configuration is in `training/config.yaml`. Key parameters:

- `base_model`: "microsoft/codebert-base"
- `max_length`: 512 (token limit)
- `batch_size`: 8 (adjust based on GPU memory)
- `learning_rate`: 2e-5
- `epochs`: 3

### Model Training

Training uses HuggingFace Trainer with:

- Early stopping based on validation F1
- Model checkpointing
- Automatic device detection (CPU/GPU/MPS)

```bash
# Train with default config
python training/train.py --config training/config.yaml

# Train with custom config
python training/train.py --config path/to/custom_config.yaml
```

### Evaluation

The evaluation script provides comprehensive metrics:

```bash
python training/evaluate.py \
    --config training/config.yaml \
    --model_path serving/model_store/best_model
```

Output includes:
- Overall accuracy, precision, recall, F1
- Per-class metrics (clean vs buggy)
- Per-project F1 scores
- QuixBugs sanity check results

### API Usage

#### Health Check

```bash
curl http://localhost:8000/health
```

#### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"code": "def buggy_function():\n    return 1 / 0"}'
```

#### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict-batch" \
     -H "Content-Type: application/json" \
     -d '["def test1(): return 1 / 0", "def test2(): return 42"]'
```

#### Model Information

```bash
curl http://localhost:8000/model-info
```

### Docker Deployment

#### Build Images

```bash
# Build API image
docker build -f docker/Dockerfile.api -t ai-code-reviewer-api .

# Build Streamlit image
docker build -f docker/Dockerfile.streamlit -t ai-code-reviewer-streamlit .
```

#### Run with Docker Compose

```bash
# Start both services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### Manual Docker Run

```bash
# Run API
docker run -p 8000:8000 -v $(pwd)/serving/model_store:/app/serving/model_store ai-code-reviewer-api

# Run Streamlit
docker run -p 8501:8501 -v $(pwd)/serving/model_store:/app/serving/model_store ai-code-reviewer-streamlit
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory During Training

- Reduce `batch_size` in `training/config.yaml`
- Use gradient accumulation: increase `grad_accum_steps`
- Use CPU training: set `device: "cpu"`

#### 2. Model Not Found

```bash
# Check if model exists
ls -la serving/model_store/best_model/

# Retrain if missing
make train
```

#### 3. API Connection Issues

```bash
# Check if API is running
curl http://localhost:8000/health

# Check logs
docker-compose logs api
```

#### 4. Dataset Download Issues

```bash
# Manual download
git clone https://github.com/soarsmu/BugsInPy data/raw/bugsinpy
git clone https://github.com/jkoppel/QuixBugs data/raw/quixbugs
```

#### 5. CUDA/GPU Issues

- Install CUDA drivers and PyTorch with CUDA support
- Or use CPU: set `device: "cpu"` in config
- For M1 Mac: set `device: "mps"`

### Performance Optimization

#### Training

- Use GPU if available
- Increase batch size for better GPU utilization
- Use mixed precision training (add to config)
- Use gradient checkpointing for large models

#### Inference

- Use batch predictions for multiple samples
- Enable model caching in production
- Use quantization for smaller model size
- Consider model distillation for faster inference

### Monitoring

#### Training Metrics

- Monitor training loss and validation metrics
- Check for overfitting (validation loss increasing)
- Monitor GPU memory usage

#### API Metrics

- Response time
- Throughput (requests/second)
- Error rates
- Model loading time

## Production Deployment

### Environment Variables

```bash
export MODEL_DIR=serving/model_store/best_model
export API_HOST=0.0.0.0
export API_PORT=8000
export LOG_LEVEL=INFO
```

### Security Considerations

- Use HTTPS in production
- Implement authentication/authorization
- Rate limiting
- Input validation
- Model versioning

### Scaling

- Use multiple API instances behind a load balancer
- Implement model caching
- Use async processing for batch predictions
- Consider model serving frameworks (TorchServe, TensorFlow Serving)

## Maintenance

### Regular Tasks

- Monitor model performance
- Retrain with new data
- Update dependencies
- Backup model artifacts
- Monitor system resources

### Model Updates

```bash
# Retrain with new data
make data
make train
make eval

# Deploy new model
docker-compose down
docker-compose up -d
```

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review logs and error messages
3. Check GitHub issues
4. Create a new issue with detailed information

## License

This project is licensed under the MIT License. See LICENSE file for details.
