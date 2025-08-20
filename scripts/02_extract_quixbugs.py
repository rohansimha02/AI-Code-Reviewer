#!/usr/bin/env python3
"""
Extract buggy and fixed code pairs from QuixBugs dataset.
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_function_name(code: str) -> str:
    """Extract function name from Python code."""
    # Look for function definitions
    func_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    match = re.search(func_pattern, code)
    if match:
        return match.group(1)
    
    # Look for class definitions
    class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    match = re.search(class_pattern, code)
    if match:
        return match.group(1)
    
    return "module"


def extract_code_from_file(file_path: Path, max_chars: int = 4000) -> Optional[str]:
    """Extract code from file with length limit."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if len(content) > max_chars:
            # Truncate but try to keep complete functions
            lines = content.split('\n')
            truncated_lines = []
            current_length = 0
            
            for line in lines:
                if current_length + len(line) + 1 > max_chars:
                    break
                truncated_lines.append(line)
                current_length += len(line) + 1
            
            return '\n'.join(truncated_lines)
        
        return content
    except Exception as e:
        logger.warning(f"Error reading file {file_path}: {e}")
        return None


def process_quixbugs_algorithm(
    algorithm_name: str, 
    raw_dir: Path, 
    max_chars: int
) -> List[Dict]:
    """Process a single QuixBugs algorithm."""
    # Check for files in both directories
    buggy_file = raw_dir / "python_programs" / f"{algorithm_name}.py"
    fixed_file = raw_dir / "correct_python_programs" / f"{algorithm_name}.py"
    
    if not buggy_file.exists():
        logger.warning(f"Buggy file not found: {buggy_file}")
        return []
    
    if not fixed_file.exists():
        logger.warning(f"Fixed file not found: {fixed_file}")
        return []
    
    # Extract code from both files
    buggy_code = extract_code_from_file(buggy_file, max_chars)
    fixed_code = extract_code_from_file(fixed_file, max_chars)
    
    if not buggy_code or not fixed_code:
        logger.warning(f"Could not extract code for {algorithm_name}")
        return []
    
    # Check if the codes are actually different
    if buggy_code.strip() == fixed_code.strip():
        logger.info(f"Buggy and fixed code are identical for {algorithm_name}, skipping")
        return []
    
    pair_id = f"quixbugs_{algorithm_name}"
    
    # Buggy version
    buggy_record = {
        'id': f"{pair_id}_buggy",
        'source': 'quixbugs',
        'project': 'quixbugs',
        'path': str(buggy_file.relative_to(raw_dir)),
        'func_name': extract_function_name(buggy_code),
        'code': buggy_code,
        'is_buggy': 1,
        'pair_id': pair_id
    }
    
    # Fixed version
    fixed_record = {
        'id': f"{pair_id}_fixed",
        'source': 'quixbugs',
        'project': 'quixbugs',
        'path': str(fixed_file.relative_to(raw_dir)),
        'func_name': extract_function_name(fixed_code),
        'code': fixed_code,
        'is_buggy': 0,
        'pair_id': pair_id
    }
    
    return [buggy_record, fixed_record]


def main():
    parser = argparse.ArgumentParser(description="Extract buggy and fixed code pairs from QuixBugs")
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to QuixBugs raw data directory")
    parser.add_argument("--out", type=str, required=True, help="Output parquet file path")
    parser.add_argument("--max_chars", type=int, default=4000, help="Maximum characters per code snippet")
    
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        logger.error(f"Raw directory does not exist: {raw_dir}")
        logger.error("Please ensure QuixBugs is properly set up.")
        return
    
    # Check for required directories
    python_programs_dir = raw_dir / "python_programs"
    correct_programs_dir = raw_dir / "correct_python_programs"
    
    if not python_programs_dir.exists():
        logger.error(f"python_programs directory not found: {python_programs_dir}")
        logger.error("Please ensure QuixBugs contains the correct directory structure")
        return
    
    if not correct_programs_dir.exists():
        logger.error(f"correct_python_programs directory not found: {correct_programs_dir}")
        logger.error("Please ensure QuixBugs contains the correct directory structure")
        return
    
    # Get all Python files from the buggy directory
    buggy_files = list(python_programs_dir.glob("*.py"))
    logger.info(f"Found {len(buggy_files)} buggy Python files")
    
    all_results = []
    
    for buggy_file in tqdm.tqdm(buggy_files, desc="Processing algorithms"):
        algorithm_name = buggy_file.stem  # Remove .py extension
        
        logger.info(f"Processing algorithm: {algorithm_name}")
        
        results = process_quixbugs_algorithm(
            algorithm_name, 
            raw_dir, 
            args.max_chars
        )
        
        all_results.extend(results)
        logger.info(f"Extracted {len(results)} code pairs from {algorithm_name}")
    
    # Create DataFrame and save
    if all_results:
        df = pd.DataFrame(all_results)
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(df)} code pairs to {output_path}")
        
        # Print summary statistics
        logger.info(f"Total pairs: {len(df) // 2}")
        logger.info(f"Buggy samples: {len(df[df['is_buggy'] == 1])}")
        logger.info(f"Clean samples: {len(df[df['is_buggy'] == 0])}")
        logger.info(f"Algorithms: {len(buggy_files)}")
    else:
        logger.warning("No code pairs extracted")


if __name__ == "__main__":
    main()
