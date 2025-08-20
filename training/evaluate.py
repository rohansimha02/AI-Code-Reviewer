#!/usr/bin/env python3
"""
Evaluation script for CodeBERT classifier.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training.datamodule import CodeDataModule
from training.utils import set_seed, compute_metrics, get_device

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_test_data(test_path: str) -> pd.DataFrame:
    """Load test dataset."""
    df = pd.read_parquet(test_path)
    logger.info(f"Loaded {len(df)} test samples from {test_path}")
    return df


def evaluate_model(model, tokenizer, test_data: pd.DataFrame, max_length: int = 512):
    """Evaluate model on test data."""
    device = next(model.parameters()).device
    
    predictions = []
    true_labels = []
    confidences = []
    
    model.eval()
    
    with torch.no_grad():
        for idx, row in test_data.iterrows():
            code = row['code']
            true_label = row['is_buggy']
            
            # Tokenize input
            inputs = tokenizer(
                code,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get prediction
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = torch.max(probabilities, dim=-1).values.item()
            
            predictions.append(predicted_class)
            true_labels.append(true_label)
            confidences.append(confidence)
    
    return predictions, true_labels, confidences


def compute_per_project_metrics(test_data: pd.DataFrame, predictions: List[int], true_labels: List[int]):
    """Compute metrics per project."""
    # Add predictions to dataframe
    test_data = test_data.copy()
    test_data['predicted'] = predictions
    test_data['true_label'] = true_labels
    
    project_metrics = []
    
    for project in test_data['project'].unique():
        project_data = test_data[test_data['project'] == project]
        
        if len(project_data) == 0:
            continue
        
        # Compute metrics for this project
        project_preds = project_data['predicted'].tolist()
        project_true = project_data['true_label'].tolist()
        
        # Calculate metrics
        accuracy = sum(1 for p, t in zip(project_preds, project_true) if p == t) / len(project_preds)
        
        # Calculate F1 score
        f1 = f1_score(project_true, project_preds, average='macro', zero_division=0)
        
        project_metrics.append({
            'project': project,
            'samples': len(project_data),
            'accuracy': accuracy,
            'f1_macro': f1,
            'buggy_samples': len(project_data[project_data['true_label'] == 1]),
            'clean_samples': len(project_data[project_data['true_label'] == 0])
        })
    
    return project_metrics


def evaluate_quixbugs(model, tokenizer, quixbugs_path: str, max_length: int = 512):
    """Evaluate model on QuixBugs dataset."""
    if not os.path.exists(quixbugs_path):
        logger.warning(f"QuixBugs evaluation file not found: {quixbugs_path}")
        return None
    
    quixbugs_data = pd.read_parquet(quixbugs_path)
    logger.info(f"Evaluating on QuixBugs: {len(quixbugs_data)} samples")
    
    predictions, true_labels, confidences = evaluate_model(
        model, tokenizer, quixbugs_data, max_length
    )
    
    # Compute metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1,
        'samples': len(quixbugs_data),
        'buggy_samples': sum(true_labels),
        'clean_samples': len(true_labels) - sum(true_labels)
    }


def print_results(predictions: List[int], true_labels: List[int], confidences: List[float], 
                 project_metrics: List[Dict], quixbugs_metrics: Dict = None):
    """Print evaluation results."""
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    
    # Overall metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro', zero_division=0
    )
    
    logger.info(f"Overall Test Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision (macro): {precision:.4f}")
    logger.info(f"  Recall (macro): {recall:.4f}")
    logger.info(f"  F1 (macro): {f1:.4f}")
    logger.info(f"  Average confidence: {sum(confidences) / len(confidences):.4f}")
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        true_labels, predictions, average=None, zero_division=0
    )
    
    logger.info(f"\nPer-class Results:")
    logger.info(f"  Clean - Precision: {precision_per_class[0]:.4f}, Recall: {recall_per_class[0]:.4f}, F1: {f1_per_class[0]:.4f}")
    logger.info(f"  Buggy - Precision: {precision_per_class[1]:.4f}, Recall: {recall_per_class[1]:.4f}, F1: {f1_per_class[1]:.4f}")
    
    # Per-project results
    logger.info(f"\nPer-project F1 Scores:")
    logger.info("-" * 40)
    for metric in sorted(project_metrics, key=lambda x: x['f1_macro'], reverse=True):
        logger.info(f"  {metric['project']:<20} F1: {metric['f1_macro']:.4f} ({metric['samples']} samples)")
    
    # QuixBugs results
    if quixbugs_metrics:
        logger.info(f"\nQuixBugs Evaluation:")
        logger.info(f"  Accuracy: {quixbugs_metrics['accuracy']:.4f}")
        logger.info(f"  F1 (macro): {quixbugs_metrics['f1_macro']:.4f}")
        logger.info(f"  Samples: {quixbugs_metrics['samples']}")
    
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate AI Code Reviewer model")
    parser.add_argument("--config", type=str, default="training/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model_path", type=str, default="serving/model_store/best_model",
                       help="Path to trained model")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed for reproducibility
    set_seed(config.get("seed", 42))
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        logger.error("Please train the model first using: make train")
        return
    
    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    
    # Move to device
    device = get_device(config.get("device", "auto"))
    model = model.to(device)
    
    # Load test data
    test_data = load_test_data(config.get("test_path", "data/processed/test.parquet"))
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    predictions, true_labels, confidences = evaluate_model(
        model, tokenizer, test_data, config.get("max_length", 512)
    )
    
    # Compute per-project metrics
    logger.info("Computing per-project metrics...")
    project_metrics = compute_per_project_metrics(test_data, predictions, true_labels)
    
    # Evaluate on QuixBugs
    quixbugs_metrics = evaluate_quixbugs(
        model, tokenizer, config.get("quixbugs_path", "data/processed/quixbugs_eval.parquet")
    )
    
    # Print results
    print_results(predictions, true_labels, confidences, project_metrics, quixbugs_metrics)
    
    # Save results
    results = {
        'test_metrics': {
            'accuracy': accuracy_score(true_labels, predictions),
            'f1_macro': f1_score(true_labels, predictions, average='macro', zero_division=0),
            'samples': len(test_data)
        },
        'project_metrics': project_metrics,
        'quixbugs_metrics': quixbugs_metrics
    }
    
    results_path = Path("training/evaluation_results.json")
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
