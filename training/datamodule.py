"""
Data module for AI Code Reviewer training.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
import random
import re

from .tokenizer import CodeTokenizer

logger = logging.getLogger(__name__)


class CodeDataset(Dataset):
    """Dataset for code classification."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512, augment: bool = False):
        self.data = pd.read_parquet(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        
    def __len__(self):
        return len(self.data)
    
    def augment_code(self, code: str) -> str:
        """Apply simple data augmentation to code."""
        if not self.augment or random.random() > 0.3:  # 30% chance of augmentation
            return code
            
        augmented = code
        
        # 1. Randomly add/remove whitespace
        if random.random() < 0.2:
            lines = augmented.split('\n')
            if len(lines) > 1:
                # Randomly add extra spaces to some lines
                for i in range(len(lines)):
                    if random.random() < 0.1:  # 10% chance per line
                        lines[i] = ' ' * random.randint(1, 4) + lines[i]
                augmented = '\n'.join(lines)
        
        # 2. Randomly add comments
        if random.random() < 0.15:
            comments = [
                "# TODO: review this code",
                "# FIXME: potential issue here", 
                "# NOTE: important implementation",
                "# BUG: needs investigation"
            ]
            comment = random.choice(comments)
            lines = augmented.split('\n')
            if lines:
                insert_pos = random.randint(0, min(len(lines), 3))
                lines.insert(insert_pos, comment)
                augmented = '\n'.join(lines)
        
        return augmented
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        code = row['code']
        if self.augment:
            code = self.augment_code(code)
        
        # Tokenize
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': row['is_buggy']
        }


class CodeDataModule:
    """Data module for code classification."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.data_collator = None
    
    def setup(self):
        """Setup datasets."""
        # Initialize tokenizer
        self.tokenizer = CodeTokenizer(self.config["base_model"], self.config["max_length"])
        
        # Create datasets
        self.train_dataset = CodeDataset(
            self.config["train_path"], 
            self.tokenizer.tokenizer, 
            self.config["max_length"],
            augment=True  # Enable augmentation for training
        )
        
        self.val_dataset = CodeDataset(
            self.config["val_path"], 
            self.tokenizer.tokenizer, 
            self.config["max_length"],
            augment=False  # No augmentation for validation
        )
        
        self.test_dataset = CodeDataset(
            self.config["test_path"], 
            self.tokenizer.tokenizer, 
            self.config["max_length"],
            augment=False  # No augmentation for testing
        )
        
        # Create data collator
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer.tokenizer)
        
        logger.info(f"Train dataset: {len(self.train_dataset)} samples")
        logger.info(f"Val dataset: {len(self.val_dataset)} samples")
        logger.info(f"Test dataset: {len(self.test_dataset)} samples")
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=0,  # Set to 0 for debugging, increase for production
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=True
        )
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics about the datasets."""
        if self.train_dataset is None:
            self.setup()
        
        stats = {}
        
        for name, dataset in [
            ("train", self.train_dataset),
            ("val", self.val_dataset),
            ("test", self.test_dataset)
        ]:
            if dataset is not None:
                data = dataset.data
                stats[name] = {
                    "total_samples": len(data),
                    "buggy_samples": len(data[data['is_buggy'] == 1]),
                    "clean_samples": len(data[data['is_buggy'] == 0]),
                    "projects": data['project'].nunique(),
                    "avg_code_length": data['code'].str.len().mean(),
                }
        
        return stats


def create_datamodule(config: Dict) -> CodeDataModule:
    """Create data module from configuration."""
    return CodeDataModule(config)


def analyze_datasets(datamodule: CodeDataModule):
    """Analyze dataset characteristics."""
    logger.info("Dataset Analysis:")
    logger.info("=" * 50)
    
    stats = datamodule.get_dataset_stats()
    
    for split_name, split_stats in stats.items():
        logger.info(f"\n{split_name.upper()} SPLIT:")
        logger.info(f"  Total samples: {split_stats['total_samples']:,}")
        logger.info(f"  Buggy samples: {split_stats['buggy_samples']:,}")
        logger.info(f"  Clean samples: {split_stats['clean_samples']:,}")
        logger.info(f"  Projects: {split_stats['projects']}")
        logger.info(f"  Average code length: {split_stats['avg_code_length']:.0f} chars")
        
        if split_stats['total_samples'] > 0:
            buggy_ratio = split_stats['buggy_samples'] / split_stats['total_samples']
            logger.info(f"  Buggy ratio: {buggy_ratio:.1%}")
    
    logger.info("=" * 50)
