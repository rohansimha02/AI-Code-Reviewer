#!/usr/bin/env python3
"""
Training script for CodeBERT classifier.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from training.datamodule import CodeDataModule
from training.utils import set_seed, compute_metrics, get_device, save_model_card, log_training_info

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure numeric values are properly typed
    config["learning_rate"] = float(config["learning_rate"])
    config["batch_size"] = int(config["batch_size"])
    config["epochs"] = int(config["epochs"])
    config["max_length"] = int(config["max_length"])
    config["eval_steps"] = int(config["eval_steps"])
    config["warmup_ratio"] = float(config["warmup_ratio"])
    config["weight_decay"] = float(config["weight_decay"])
    config["grad_accum_steps"] = int(config["grad_accum_steps"])
    
    return config


def create_trainer(
    model: AutoModelForSequenceClassification,
    datamodule: CodeDataModule,
    config: Dict,
    save_dir: Path,
) -> Trainer:
    """Create and configure the Trainer."""
    
    training_args = TrainingArguments(
        output_dir=str(save_dir),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        gradient_accumulation_steps=config["grad_accum_steps"],
        max_grad_norm=1.0,  # Add gradient clipping
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        evaluation_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["eval_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",  # Changed from eval_f1 to eval_f1_macro
        greater_is_better=True,
        logging_dir=str(save_dir / "logs"),
        logging_steps=100,
        report_to=None,  # Disable wandb/tensorboard
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datamodule.train_dataset,
        eval_dataset=datamodule.val_dataset,
        data_collator=datamodule.data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    return trainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train CodeBERT classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config["seed"])
    
    # Log configuration
    log_training_info(config)
    
    # Get device
    device = get_device(config["device"])
    logger.info(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    logger.info(f"Initialized tokenizer for {config['base_model']}")
    logger.info(f"Max length: {config['max_length']}")
    
    # Setup datasets
    logger.info("Setting up datasets...")
    datamodule = CodeDataModule(config)
    datamodule.setup()
    
    # Analyze datasets
    from training.datamodule import analyze_datasets
    analyze_datasets(datamodule)
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"],
        num_labels=config["num_labels"],
    )
    
    # Add class weights if specified
    if config.get("use_class_weights", False):
        class_weights = torch.tensor(config.get("class_weights", [1.0, 1.0]), dtype=torch.float32)
        # Apply class weights to the classifier output layer
        if hasattr(model.classifier, 'out_proj') and hasattr(model.classifier.out_proj, 'bias'):
            model.classifier.out_proj.bias.data = torch.log(class_weights).to(model.device)
        logger.info(f"Applied class weights: {class_weights}")
    
    # Move model to device (but keep on CPU if device is cpu)
    if device.type == "cpu":
        model = model.to("cpu")
    else:
        model = model.to(device)
    
    logger.info("Starting training...")
    
    # Create trainer
    save_dir = Path(config.get("save_dir", "serving/model_store/best_model"))
    os.makedirs(save_dir, exist_ok=True)

    trainer = create_trainer(
        model, 
        datamodule, 
        config, 
        save_dir
    )
    
    # Train the model
    train_result = trainer.train()
    
    # Save the model
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # Save model card
    save_model_card(save_dir, config, train_result)
    
    logger.info(f"Training completed! Model saved to {save_dir}")
    logger.info(f"Final training loss: {train_result.training_loss:.4f}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(datamodule.test_dataset)
    logger.info(f"Test accuracy: {test_results['eval_accuracy']:.4f}")
    
    # Check if F1 score is available
    if 'eval_f1_macro' in test_results:
        logger.info(f"Test F1: {test_results['eval_f1_macro']:.4f}")
    else:
        logger.info("F1 score not available in test results")
        logger.info(f"Available metrics: {list(test_results.keys())}")


if __name__ == "__main__":
    main()
