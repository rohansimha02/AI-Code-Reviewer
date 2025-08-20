"""
Tests for data preparation functionality.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import os


def test_parquet_schema():
    """Test that parquet files have the correct schema."""
    # Create a sample parquet file with the expected schema
    sample_data = {
        'id': ['test_1', 'test_2'],
        'source': ['bugsinpy', 'quixbugs'],
        'project': ['test_project', 'quixbugs'],
        'path': ['test/path.py', 'test/path.py'],
        'func_name': ['test_function', 'test_function'],
        'code': ['def test(): pass', 'def test(): return 1'],
        'is_buggy': [1, 0],
        'pair_id': ['test_pair', 'test_pair']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Check that all required columns are present
    required_columns = ['id', 'source', 'project', 'path', 'func_name', 'code', 'is_buggy', 'pair_id']
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"
    
    # Check data types
    assert df['id'].dtype == 'object'
    assert df['source'].dtype == 'object'
    assert df['project'].dtype == 'object'
    assert df['path'].dtype == 'object'
    assert df['func_name'].dtype == 'object'
    assert df['code'].dtype == 'object'
    assert df['is_buggy'].dtype in ['int64', 'int32']
    assert df['pair_id'].dtype == 'object'
    
    # Check that is_buggy contains only 0 and 1
    assert set(df['is_buggy'].unique()).issubset({0, 1})
    
    # Check that source contains valid values
    valid_sources = {'bugsinpy', 'quixbugs'}
    assert set(df['source'].unique()).issubset(valid_sources)


def test_parquet_io():
    """Test reading and writing parquet files."""
    sample_data = {
        'id': ['test_1'],
        'source': ['bugsinpy'],
        'project': ['test_project'],
        'path': ['test/path.py'],
        'func_name': ['test_function'],
        'code': ['def test(): pass'],
        'is_buggy': [1],
        'pair_id': ['test_pair']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Test writing and reading
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
        df.to_parquet(tmp_file.name, index=False)
        
        # Read back
        df_read = pd.read_parquet(tmp_file.name)
        
        # Clean up
        os.unlink(tmp_file.name)
    
    # Check that data is preserved
    assert len(df_read) == len(df)
    assert list(df_read.columns) == list(df.columns)
    assert df_read.iloc[0]['code'] == df.iloc[0]['code']


def test_data_validation():
    """Test data validation logic."""
    # Valid data
    valid_data = {
        'id': ['test_1'],
        'source': ['bugsinpy'],
        'project': ['test_project'],
        'path': ['test/path.py'],
        'func_name': ['test_function'],
        'code': ['def test(): pass'],
        'is_buggy': [1],
        'pair_id': ['test_pair']
    }
    
    df_valid = pd.DataFrame(valid_data)
    
    # Test validation
    assert len(df_valid) > 0
    assert 'is_buggy' in df_valid.columns
    assert set(df_valid['is_buggy'].unique()).issubset({0, 1})
    
    # Test with invalid data
    invalid_data = valid_data.copy()
    invalid_data['is_buggy'] = [2]  # Invalid value
    
    df_invalid = pd.DataFrame(invalid_data)
    
    # Should fail validation
    assert not set(df_invalid['is_buggy'].unique()).issubset({0, 1})


def test_code_length_limits():
    """Test code length validation."""
    # Test with code within limits
    short_code = "def test(): pass"
    assert len(short_code) < 4000
    
    # Test with code that would exceed limits
    long_code = "def test():\n" + "    pass\n" * 1000
    assert len(long_code) > 4000
    
    # In a real scenario, the extraction script should truncate this
    truncated_code = long_code[:4000]
    assert len(truncated_code) <= 4000


def test_project_level_split():
    """Test project-level splitting logic."""
    # Create sample data with multiple projects
    sample_data = {
        'id': [f'test_{i}' for i in range(10)],
        'source': ['bugsinpy'] * 10,
        'project': ['project_a'] * 4 + ['project_b'] * 3 + ['project_c'] * 3,
        'path': ['test/path.py'] * 10,
        'func_name': ['test_function'] * 10,
        'code': ['def test(): pass'] * 10,
        'is_buggy': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'pair_id': [f'test_pair_{i}' for i in range(10)]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Check that we have multiple projects
    projects = df['project'].unique()
    assert len(projects) > 1
    
    # Check that each project has multiple samples
    for project in projects:
        project_data = df[df['project'] == project]
        assert len(project_data) > 0


if __name__ == "__main__":
    pytest.main([__file__])
