"""
Pytest configuration for the house price prediction system tests.
"""
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
test_dir = Path(__file__).parent
src_dir = test_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Set up test environment
import pytest


@pytest.fixture(scope="session")
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.randn(100) * 1000 + 50000
    })


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    from unittest.mock import Mock
    import numpy as np
    
    model = Mock()
    model.predict.return_value = np.random.randn(100) * 1000 + 50000
    return model


def pytest_configure(config):
    """Configure pytest."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test items."""
    # Add unit marker to all tests by default
    for item in items:
        if not any(mark.name in ['integration', 'slow'] for mark in item.iter_markers()):
            item.add_marker(pytest.mark.unit)