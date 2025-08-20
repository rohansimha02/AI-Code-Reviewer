# AI Code Reviewer

A production-style repository that fine-tunes a pretrained CodeBERT classifier (PyTorch + Hugging Face) to detect buggy vs clean code. The system is trained primarily on BugsInPy with QuixBugs as a small validation set, providing a minimal but polished FastAPI inference service and a Streamlit demo where users can paste Python code and get predictions.

## Dataset

- **BugsInPy**: Main training dataset containing real-world Python bugs from popular projects
- **QuixBugs**: Small validation set for sanity checks (not mixed into train/val/test splits)

## Prerequisites

This project assumes you have the BugsInPy and QuixBugs data organized in the following structure:

```
project_root/
├── data/
│   └── raw/
│       ├── bugsinpy/              # BugsInPy data
│       │   ├── projects/          # BugsinPy projects
│       │   └── framework/         # BugsInPy framework tools
│       └── quixbugs/              # QuixBugs data
│           ├── python_programs/   # QuixBugs buggy programs
│           ├── correct_python_programs/  # QuixBugs fixed programs
│           ├── json_testcases/    # QuixBugs test data
│           └── python_testcases/  # QuixBugs test cases
├── scripts/                       # Data processing scripts
├── training/                      # Model training code
├── serving/                       # FastAPI inference service
├── demo/                          # Streamlit demo
└── ...                           # Other project files
```

## Quick Start

### Setup
```bash
# Ensure you're in the project root directory
# Setup virtual environment and install dependencies
make setup

# Setup data directories (uses existing data in data/raw/)
./scripts/99_download_samples.sh
```

### Data Preparation
```bash
make data
```

### Training
```bash
make train
```

### Evaluation
```bash
make eval
```

### Serve API
```bash
make serve
```

### Launch Demo
```bash
make demo
```

### Run Tests
```bash
make test
```

## API Usage

The FastAPI service runs on `http://localhost:8000` and provides:

- `GET /health` - Health check endpoint
- `POST /predict` - Code prediction endpoint

### Example curl request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"code": "def buggy_function():\n    return 1 / 0"}'
```

Response:
```json
{
  "label": "buggy",
  "score": 0.95
}
```

## Project Structure

```
project_root/
├── data/               # Data processing pipeline
├── scripts/            # Data extraction and preprocessing
├── training/           # Model training and evaluation
├── models/             # Model definitions
├── serving/            # FastAPI inference service
├── demo/               # Streamlit demo application
├── tests/              # Test suite
└── docker/             # Docker configurations
```

## Data Processing

The system processes two main datasets:

### BugsInPy
- Extracts buggy and fixed code from git patches and checkout scripts
- Uses project-level splits to prevent data leakage
- Processes bug info files to extract metadata

### QuixBugs
- Pairs buggy and fixed Python programs
- Compares files to ensure they're actually different
- Extracts function names and metadata

## Reproducibility

- All random seeds are fixed (seed=42)
- Requirements are pinned to specific versions
- Project-level splits prevent data leakage
- Comprehensive logging and metrics tracking

## Limitations

- Small dataset size (intended as fine-tuning demo)
- Binary classification only (buggy vs clean)
- Python code only
- Function-level analysis (no file-level or line-level granularity)

## Model Details

- **Base Model**: microsoft/codebert-base
- **Task**: Sequence classification (buggy vs clean)
- **Input**: Python function snippets (max 512 tokens)
- **Output**: Binary classification with confidence scores

## Metrics

- Accuracy, Precision, Recall, F1 (macro and per-class)
- Per-project F1 on held-out projects
- QuixBugs sanity check F1
