#!/usr/bin/env python3
"""
Display dataset statistics in a readable format.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_statistics(stats_file: Path) -> List[Dict]:
    """Load statistics from JSON file."""
    if not stats_file.exists():
        logger.error(f"Statistics file not found: {stats_file}")
        return []
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    return stats


def print_statistics_table(stats):
    """Print a formatted table of dataset statistics."""
    print("=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    print(f"{'Split':<15} {'Samples':<10} {'Buggy':<8} {'Clean':<8} {'Ratio':<8} {'Projects':<10} {'Avg Len':<10}")
    print("-" * 80)
    
    for split_name, split_stats in stats.items():
        if split_name == "quixbugs":
            split_name = "quixbugs_eval"
            projects = 1  # QuixBugs is treated as one project
        else:
            projects = split_stats['projects']
        
        samples = split_stats['samples']
        buggy = split_stats['buggy']
        clean = split_stats['clean']
        ratio = buggy / samples if samples > 0 else 0
        avg_len = split_stats['avg_length']
        
        print(f"{split_name:<15} {samples:<10} {buggy:<8} {clean:<8} {ratio:<8.3f} {projects:<10} {avg_len:<10.0f}")
    
    print("-" * 80)
    print()


def print_detailed_statistics(stats: List[Dict]):
    """Print detailed statistics for each split."""
    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)
    
    for stat in stats:
        split_name = stat['split']
        print(f"\n{split_name.upper()} SPLIT:")
        print(f"  Total samples: {stat['total_samples']:,}")
        print(f"  Buggy samples: {stat['buggy_samples']:,} ({stat['buggy_ratio']:.1%})")
        print(f"  Clean samples: {stat['clean_samples']:,} ({1-stat['buggy_ratio']:.1%})")
        print(f"  Projects: {stat['projects']}")
        print(f"  Code length statistics:")
        print(f"    Average: {stat['avg_code_length']:.0f} characters")
        print(f"    Median: {stat['median_code_length']:.0f} characters")
        print(f"    Min: {stat['min_code_length']} characters")
        print(f"    Max: {stat['max_code_length']} characters")


def print_code_length_distribution(stats: List[Dict]):
    """Print code length distribution analysis."""
    print("\n" + "="*80)
    print("CODE LENGTH ANALYSIS")
    print("="*80)
    
    for stat in stats:
        split_name = stat['split']
        avg_len = stat['avg_code_length']
        median_len = stat['median_code_length']
        min_len = stat['min_code_length']
        max_len = stat['max_code_length']
        
        print(f"\n{split_name.upper()}:")
        print(f"  Average length: {avg_len:.0f} characters")
        print(f"  Median length: {median_len:.0f} characters")
        print(f"  Range: {min_len} - {max_len} characters")
        
        if avg_len > 0:
            print(f"  Length distribution:")
            if avg_len < 500:
                print(f"    Short code (< 500 chars): Likely function-level snippets")
            elif avg_len < 1000:
                print(f"    Medium code (500-1000 chars): Function or small module")
            else:
                print(f"    Long code (> 1000 chars): Module or file-level code")


def print_class_balance_analysis(stats: List[Dict]):
    """Print class balance analysis."""
    print("\n" + "="*80)
    print("CLASS BALANCE ANALYSIS")
    print("="*80)
    
    for stat in stats:
        split_name = stat['split']
        buggy_ratio = stat['buggy_ratio']
        
        print(f"\n{split_name.upper()}:")
        print(f"  Buggy ratio: {buggy_ratio:.1%}")
        
        if 0.45 <= buggy_ratio <= 0.55:
            print(f"  Status: Well balanced")
        elif 0.4 <= buggy_ratio <= 0.6:
            print(f"  Status: Moderately balanced")
        elif 0.3 <= buggy_ratio <= 0.7:
            print(f"  Status: Somewhat imbalanced")
        else:
            print(f"  Status: Highly imbalanced (consider class weights)")


def print_summary(stats):
    """Print a summary of the dataset."""
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Calculate totals (excluding quixbugs)
    total_samples = sum(split_stats['samples'] for split_name, split_stats in stats.items() if split_name != 'quixbugs')
    total_buggy = sum(split_stats['buggy'] for split_name, split_stats in stats.items() if split_name != 'quixbugs')
    total_clean = sum(split_stats['clean'] for split_name, split_stats in stats.items() if split_name != 'quixbugs')
    
    print(f"Total training samples (train + val + test): {total_samples:,}")
    print(f"Total buggy samples: {total_buggy}")
    print(f"Total clean samples: {total_clean}")
    print(f"Overall buggy ratio: {(total_buggy/total_samples)*100:.1f}%")
    print()
    
    if 'quixbugs' in stats:
        quixbugs_stats = stats['quixbugs']
        print(f"QuixBugs evaluation set: {quixbugs_stats['samples']} samples")
        print(f"  Buggy: {quixbugs_stats['buggy']}, Clean: {quixbugs_stats['clean']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Display dataset statistics")
    parser.add_argument("--stats_file", type=str, default="data/processed/stats.json",
                       help="Path to statistics JSON file")
    parser.add_argument("--detailed", action="store_true", help="Show detailed statistics")
    parser.add_argument("--length_analysis", action="store_true", help="Show code length analysis")
    parser.add_argument("--balance_analysis", action="store_true", help="Show class balance analysis")
    
    args = parser.parse_args()
    
    stats_file = Path(args.stats_file)
    if not stats_file.exists():
        logger.error(f"Statistics file not found: {stats_file}")
        logger.error("Please run the merge and split script first to generate statistics.")
        return
    
    # Load statistics
    stats = load_statistics(stats_file)
    
    if not stats:
        logger.error("No statistics found in the file")
        return
    
    # Print basic table
    print_statistics_table(stats)
    
    # Print additional analyses if requested
    if args.detailed:
        print_detailed_statistics(stats)
    
    if args.length_analysis:
        print_code_length_distribution(stats)
    
    if args.balance_analysis:
        print_class_balance_analysis(stats)
    
    # Print summary
    print_summary(stats)
    

if __name__ == "__main__":
    main()
