#!/usr/bin/env python3
"""
Merge synthetic buggy variants with existing training data.
"""

import argparse
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Merge synthetic bugs with training data")
    parser.add_argument("--train", required=True, help="Original training data")
    parser.add_argument("--synthetic", required=True, help="Synthetic buggy variants")
    parser.add_argument("--output", required=True, help="Output merged training data")
    
    args = parser.parse_args()
    
    # Load original training data
    logger.info(f"Loading original training data from {args.train}")
    train_df = pd.read_parquet(args.train)
    
    # Load synthetic bugs
    logger.info(f"Loading synthetic bugs from {args.synthetic}")
    synthetic_df = pd.read_parquet(args.synthetic)
    
    # Merge datasets
    logger.info("Merging datasets...")
    merged_df = pd.concat([train_df, synthetic_df], ignore_index=True)
    
    # Shuffle the data
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save merged dataset
    logger.info(f"Saving merged dataset to {args.output}")
    merged_df.to_parquet(args.output, index=False)
    
    # Print statistics
    print(f"\nDataset Merge Results:")
    print(f"  Original training samples: {len(train_df)}")
    print(f"  Synthetic buggy samples: {len(synthetic_df)}")
    print(f"  Total merged samples: {len(merged_df)}")
    print(f"  Buggy ratio: {(len(merged_df[merged_df['is_buggy'] == 1]) / len(merged_df)) * 100:.1f}%")
    
    # Print class distribution
    buggy_count = len(merged_df[merged_df['is_buggy'] == 1])
    clean_count = len(merged_df[merged_df['is_buggy'] == 0])
    print(f"  Buggy samples: {buggy_count}")
    print(f"  Clean samples: {clean_count}")

if __name__ == "__main__":
    main()
