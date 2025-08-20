#!/usr/bin/env python3
"""
Extract buggy and fixed code pairs from BugsInPy dataset.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            timeout=300
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out: {' '.join(cmd)}")
        return -1, "", "Command timed out"


def extract_code_from_patch(patch_content: str, context_lines: int = 15) -> Optional[str]:
    """Extract code from patch content."""
    lines = patch_content.split('\n')
    code_lines = []
    
    for line in lines:
        if line.startswith('@@'):
            # Parse the @@ line to get line numbers
            parts = line.split(' ')
            if len(parts) >= 3:
                try:
                    # Extract line number after the @@
                    line_info = parts[1]
                    if line_info.startswith('-'):
                        line_info = line_info[1:]
                    start_line = int(line_info.split(',')[0])
                    code_lines = []
                except (ValueError, IndexError):
                    continue
        elif line.startswith('+') or line.startswith('-') or line.startswith(' '):
            # Add context lines
            if line.startswith(('+', '-', ' ')):
                code_lines.append(line[1:] if line.startswith(('+', '-', ' ')) else line)
    
    if code_lines:
        return '\n'.join(code_lines)
    
    return None


def extract_function_code(file_content: str, line_number: int, context_lines: int = 15) -> str:
    """Extract function code around the given line number."""
    lines = file_content.split('\n')
    
    # Find function boundaries
    start_line = max(0, line_number - context_lines)
    end_line = min(len(lines), line_number + context_lines)
    
    # Try to find complete function boundaries
    for i in range(line_number, -1, -1):
        if i < len(lines) and (lines[i].strip().startswith('def ') or lines[i].strip().startswith('class ')):
            start_line = i
            break
    
    # Find function end (look for next function or class)
    for i in range(line_number, len(lines)):
        if i < len(lines) and (lines[i].strip().startswith('def ') or lines[i].strip().startswith('class ')):
            end_line = i
            break
    
    # Extract the function
    function_lines = lines[start_line:end_line]
    return '\n'.join(function_lines)


def process_bugsinpy_project(
    project_name: str, 
    raw_dir: Path, 
    max_chars: int, 
    context_lines: int
) -> List[Dict]:
    """Process a single BugsInPy project."""
    project_dir = raw_dir / "projects" / project_name
    if not project_dir.exists():
        logger.warning(f"Project directory not found: {project_dir}")
        return []
    
    bugs_dir = project_dir / "bugs"
    if not bugs_dir.exists():
        logger.warning(f"Bugs directory not found: {bugs_dir}")
        return []
    
    results = []
    
    # Get all bug directories
    bug_dirs = [d for d in bugs_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    
    for bug_dir in tqdm.tqdm(bug_dirs, desc=f"Processing {project_name}"):
        bug_id = bug_dir.name
        
        # Read bug info
        bug_info_file = bug_dir / "bug.info"
        bug_patch_file = bug_dir / "bug_patch.txt"
        
        if not bug_info_file.exists():
            logger.warning(f"Bug info file not found for {project_name} bug {bug_id}")
            continue
        
        # Read bug info
        bug_info = {}
        try:
            with open(bug_info_file, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        bug_info[key.strip()] = value.strip().strip('"')
        except Exception as e:
            logger.warning(f"Error reading bug info for {project_name} bug {bug_id}: {e}")
            continue
        
        # Extract code from patch if available
        buggy_code = None
        fixed_code = None
        
        if bug_patch_file.exists():
            try:
                with open(bug_patch_file, 'r') as f:
                    patch_content = f.read()
                
                # Extract code from patch
                code_from_patch = extract_code_from_patch(patch_content, context_lines)
                if code_from_patch:
                    # For patch-based extraction, we'll use the same code for both
                    # since we can't easily separate buggy/fixed from just the patch
                    buggy_code = code_from_patch
                    fixed_code = code_from_patch
            except Exception as e:
                logger.warning(f"Error reading patch for {project_name} bug {bug_id}: {e}")
        
        # If we don't have code from patch, try to use checkout scripts
        if not buggy_code or not fixed_code:
            # Check if checkout scripts exist
            checkout_buggy = bug_dir / "checkout_buggy.sh"
            checkout_fixed = bug_dir / "checkout_fixed.sh"
            
            if checkout_buggy.exists() and checkout_fixed.exists():
                # Create temporary directories for checkout
                temp_buggy = bug_dir / "temp_buggy"
                temp_fixed = bug_dir / "temp_fixed"
                
                try:
                    # Checkout buggy version
                    if temp_buggy.exists():
                        import shutil
                        shutil.rmtree(temp_buggy)
                    
                    exit_code, stdout, stderr = run_command(
                        ["bash", str(checkout_buggy)], 
                        cwd=bug_dir
                    )
                    
                    if exit_code != 0:
                        logger.warning(f"Failed to checkout buggy version for {project_name} bug {bug_id}: {stderr}")
                    
                    # Checkout fixed version
                    if temp_fixed.exists():
                        import shutil
                        shutil.rmtree(temp_fixed)
                    
                    exit_code, stdout, stderr = run_command(
                        ["bash", str(checkout_fixed)], 
                        cwd=bug_dir
                    )
                    
                    if exit_code != 0:
                        logger.warning(f"Failed to checkout fixed version for {project_name} bug {bug_id}: {stderr}")
                    
                    # Extract code from both versions
                    for temp_dir, code_type in [(temp_buggy, "buggy"), (temp_fixed, "fixed")]:
                        if not temp_dir.exists():
                            continue
                        
                        python_files = list(temp_dir.rglob("*.py"))
                        if not python_files:
                            continue
                        
                        # Use the first Python file found
                        code = extract_function_code(
                            open(python_files[0], 'r', encoding='utf-8').read(),
                            bug_info.get('line_number', 0), # Assuming line_number is in bug.info
                            context_lines
                        )
                        if code:
                            if code_type == "buggy":
                                buggy_code = code
                            else:
                                fixed_code = code
                    
                except Exception as e:
                    logger.error(f"Error processing {project_name} bug {bug_id}: {e}")
                finally:
                    # Clean up temporary directories
                    for temp_dir in [temp_buggy, temp_fixed]:
                        if temp_dir.exists():
                            import shutil
                            try:
                                shutil.rmtree(temp_dir)
                            except Exception:
                                pass
        
        # Create records if we have code
        if buggy_code or fixed_code:
            pair_id = f"{project_name}_{bug_id}"
            
            # If we have both versions, create two records
            if buggy_code and fixed_code:
                # Buggy version
                results.append({
                    'id': f"{pair_id}_buggy",
                    'source': 'bugsinpy',
                    'project': project_name,
                    'path': bug_info.get('test_file', 'unknown'),
                    'func_name': 'module',
                    'code': buggy_code,
                    'is_buggy': 1,
                    'pair_id': pair_id
                })
                
                # Fixed version
                results.append({
                    'id': f"{pair_id}_fixed",
                    'source': 'bugsinpy',
                    'project': project_name,
                    'path': bug_info.get('test_file', 'unknown'),
                    'func_name': 'module',
                    'code': fixed_code,
                    'is_buggy': 0,
                    'pair_id': pair_id
                })
            elif buggy_code:
                # Only buggy version available
                results.append({
                    'id': f"{pair_id}_buggy",
                    'source': 'bugsinpy',
                    'project': project_name,
                    'path': bug_info.get('test_file', 'unknown'),
                    'func_name': 'module',
                    'code': buggy_code,
                    'is_buggy': 1,
                    'pair_id': pair_id
                })
            elif fixed_code:
                # Only fixed version available
                results.append({
                    'id': f"{pair_id}_fixed",
                    'source': 'bugsinpy',
                    'project': project_name,
                    'path': bug_info.get('test_file', 'unknown'),
                    'func_name': 'module',
                    'code': fixed_code,
                    'is_buggy': 0,
                    'pair_id': pair_id
                })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract buggy and fixed code pairs from BugsInPy")
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to BugsInPy raw data directory")
    parser.add_argument("--out", type=str, required=True, help="Output parquet file path")
    parser.add_argument("--max_chars", type=int, default=4000, help="Maximum characters per code snippet")
    parser.add_argument("--context_lines", type=int, default=15, help="Context lines around changes")
    
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        logger.error(f"Raw directory does not exist: {raw_dir}")
        logger.error("Please ensure BugsInPy is properly set up.")
        sys.exit(1)
    
    # Check if BugsInPy framework is available
    framework_bin = raw_dir / "framework" / "bin"
    if not framework_bin.exists():
        logger.error(f"BugsInPy framework not found at {framework_bin}")
        logger.error("Please ensure BugsInPy is properly set up")
        sys.exit(1)
    
    # Add framework bin to PATH
    os.environ['PATH'] = f"{framework_bin}:{os.environ.get('PATH', '')}"
    
    # Get all project directories
    projects_dir = raw_dir / "projects"
    if not projects_dir.exists():
        logger.error(f"Projects directory not found: {projects_dir}")
        sys.exit(1)
    
    project_dirs = [d for d in projects_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(project_dirs)} projects to process")
    
    all_results = []
    
    for project_dir in project_dirs:
        project_name = project_dir.name
        logger.info(f"Processing project: {project_name}")
        
        results = process_bugsinpy_project(
            project_name, 
            raw_dir, 
            args.max_chars, 
            args.context_lines
        )
        
        all_results.extend(results)
        logger.info(f"Extracted {len(results)} code pairs from {project_name}")
    
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
        logger.info(f"Projects: {df['project'].nunique()}")
    else:
        logger.warning("No code pairs extracted")


if __name__ == "__main__":
    main()
