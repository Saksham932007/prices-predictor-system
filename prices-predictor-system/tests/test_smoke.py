"""
Smoke tests to ensure basic functionality works.
"""
import pytest
import pandas as pd
import numpy as np


def test_pandas_import():
    """Test that pandas can be imported and used."""
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    assert len(df) == 3
    assert list(df.columns) == ['a', 'b']


def test_numpy_import():
    """Test that numpy can be imported and used."""
    arr = np.array([1, 2, 3])
    assert len(arr) == 3
    assert arr.sum() == 6


def test_basic_math():
    """Test basic mathematical operations."""
    assert 2 + 2 == 4
    assert 3 * 3 == 9
    assert 10 / 2 == 5


def test_dataframe_operations():
    """Test basic DataFrame operations."""
    df = pd.DataFrame({
        'price': [100000, 150000, 200000],
        'sqft': [1000, 1500, 2000],
        'bedrooms': [2, 3, 4]
    })
    
    assert len(df) == 3
    assert df['price'].mean() == 150000
    assert df['sqft'].max() == 2000


def test_numpy_operations():
    """Test basic numpy operations."""
    data = np.random.seed(42)
    arr = np.random.randn(100)
    
    assert len(arr) == 100
    assert isinstance(arr.mean(), float)
    assert isinstance(arr.std(), float)


@pytest.mark.unit
def test_data_types():
    """Test data type handling."""
    df = pd.DataFrame({
        'int_col': [1, 2, 3],
        'float_col': [1.1, 2.2, 3.3],
        'str_col': ['a', 'b', 'c']
    })
    
    assert df['int_col'].dtype == 'int64'
    assert df['float_col'].dtype == 'float64'
    assert df['str_col'].dtype == 'object'


def test_project_structure():
    """Test that the project structure exists."""
    import os
    
    # Check that we're in the right directory structure
    current_dir = os.path.dirname(__file__)
    assert 'tests' in current_dir
    
    # Check that parent directory exists
    parent_dir = os.path.dirname(current_dir)
    assert os.path.exists(parent_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])