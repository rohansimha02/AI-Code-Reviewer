#!/usr/bin/env python3
"""
Merge BugsInPy and QuixBugs datasets and create project-level splits.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load dataset from parquet file."""
    if not file_path.exists():
        logger.warning(f"Dataset file not found: {file_path}")
        return pd.DataFrame()
    
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded {len(df)} samples from {file_path}")
    return df


def create_project_level_split(
    df: pd.DataFrame, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.1, 
    test_ratio: float = 0.2,
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """Create project-level splits to prevent data leakage."""
    # Get unique projects
    projects = df['project'].unique()
    logger.info(f"Found {len(projects)} unique projects")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Shuffle projects
    projects_shuffled = np.random.permutation(projects)
    
    # Calculate split indices
    n_projects = len(projects_shuffled)
    train_end = int(n_projects * train_ratio)
    val_end = train_end + int(n_projects * val_ratio)
    
    # Split projects
    train_projects = projects_shuffled[:train_end]
    val_projects = projects_shuffled[train_end:val_end]
    test_projects = projects_shuffled[val_end:]
    
    logger.info(f"Train projects: {len(train_projects)}")
    logger.info(f"Val projects: {len(val_projects)}")
    logger.info(f"Test projects: {len(test_projects)}")
    
    # Create splits
    train_df = df[df['project'].isin(train_projects)].copy()
    val_df = df[df['project'].isin(val_projects)].copy()
    test_df = df[df['project'].isin(test_projects)].copy()
    
    # Reset indices
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'train_projects': train_projects,
        'val_projects': val_projects,
        'test_projects': test_projects
    }


def compute_statistics(df: pd.DataFrame, split_name: str) -> Dict:
    """Compute statistics for a dataset split."""
    stats = {
        'split': split_name,
        'total_samples': len(df),
        'buggy_samples': len(df[df['is_buggy'] == 1]),
        'clean_samples': len(df[df['is_buggy'] == 0]),
        'projects': df['project'].nunique(),
        'avg_code_length': df['code'].str.len().mean(),
        'median_code_length': df['code'].str.len().median(),
        'min_code_length': df['code'].str.len().min(),
        'max_code_length': df['code'].str.len().max(),
    }
    
    if len(df) > 0:
        stats['buggy_ratio'] = stats['buggy_samples'] / stats['total_samples']
    else:
        stats['buggy_ratio'] = 0.0
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Merge and split datasets")
    parser.add_argument("--bugsinpy_path", type=str, default="data/interim/bugsinpy_pairs.parquet", 
                       help="Path to BugsInPy parquet file")
    parser.add_argument("--quixbugs_path", type=str, default="data/interim/quixbugs_pairs.parquet",
                       help="Path to QuixBugs parquet file")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                       help="Output directory for processed datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test set ratio")
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        logger.error(f"Ratios must sum to 1.0, got {total_ratio}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    bugsinpy_df = load_dataset(Path(args.bugsinpy_path))
    quixbugs_df = load_dataset(Path(args.quixbugs_path))
    
    if bugsinpy_df.empty:
        logger.error("No BugsInPy data found. Please run the BugsInPy extraction script first.")
        return
    
    # Filter out QuixBugs from BugsInPy (in case there's overlap)
    bugsinpy_df = bugsinpy_df[bugsinpy_df['source'] == 'bugsinpy'].copy()
    
    # Create project-level splits for BugsInPy
    logger.info("Creating project-level splits for BugsInPy...")
    
    # Get unique projects
    projects = bugsinpy_df['project'].unique()
    logger.info(f"Found {len(projects)} unique projects")
    
    # Sort projects for reproducible splits
    projects = sorted(projects)
    
    # Create splits: 70% train, 15% val, 15% test (instead of 70/10/20)
    n_projects = len(projects)
    n_train = int(0.7 * n_projects)
    n_val = int(0.15 * n_projects)
    n_test = n_projects - n_train - n_val
    
    train_projects = projects[:n_train]
    val_projects = projects[n_train:n_train + n_val]
    test_projects = projects[n_train + n_val:]
    
    logger.info(f"Train projects: {len(train_projects)}")
    logger.info(f"Val projects: {len(val_projects)}")
    logger.info(f"Test projects: {len(test_projects)}")
    
    # Create splits
    train_df = bugsinpy_df[bugsinpy_df['project'].isin(train_projects)].copy()
    val_df = bugsinpy_df[bugsinpy_df['project'].isin(val_projects)].copy()
    test_df = bugsinpy_df[bugsinpy_df['project'].isin(test_projects)].copy()
    
    # Reset indices
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    
    # Save splits
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)
    
    # Save QuixBugs separately for sanity checks
    if not quixbugs_df.empty:
        quixbugs_df.to_parquet(output_dir / "quixbugs_eval.parquet", index=False)
        logger.info(f"Saved QuixBugs evaluation set with {len(quixbugs_df)} samples")
    
    # Compute and save statistics
    all_stats = {
        "train": {
            "samples": int(len(train_df)),
            "buggy": int(len(train_df[train_df['is_buggy'] == 1])),
            "clean": int(len(train_df[train_df['is_buggy'] == 0])),
            "projects": int(len(train_df['project'].unique())),
            "avg_length": float(train_df['code'].str.len().mean())
        },
        "val": {
            "samples": int(len(val_df)),
            "buggy": int(len(val_df[val_df['is_buggy'] == 1])),
            "clean": int(len(val_df[val_df['is_buggy'] == 0])),
            "projects": int(len(val_df['project'].unique())),
            "avg_length": float(val_df['code'].str.len().mean())
        },
        "test": {
            "samples": int(len(test_df)),
            "buggy": int(len(test_df[test_df['is_buggy'] == 1])),
            "clean": int(len(test_df[test_df['is_buggy'] == 0])),
            "projects": int(len(test_df['project'].unique())),
            "avg_length": float(test_df['code'].str.len().mean())
        },
        "quixbugs": {
            "samples": int(len(quixbugs_df)),
            "buggy": int(len(quixbugs_df[quixbugs_df['is_buggy'] == 1])),
            "clean": int(len(quixbugs_df[quixbugs_df['is_buggy'] == 0])),
            "avg_length": float(quixbugs_df['code'].str.len().mean())
        }
    }
    
    # Save statistics
    with open(output_dir / "stats.json", 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    # Print project distribution
    logger.info("\nProject distribution:")
    logger.info(f"Train projects: {sorted(train_projects)}")
    logger.info(f"Val projects: {sorted(val_projects)}")
    logger.info(f"Test projects: {sorted(test_projects)}")
    
    logger.info(f"\nAll datasets saved to {output_dir}")
    logger.info(f"Statistics saved to {output_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
