"""
Basic test to verify test runner functionality.
This test should always pass in any Python environment.
"""
import sys
import os


def test_python_version():
    """Test that Python version is supported."""
    assert sys.version_info >= (3, 8), f"Python version {sys.version} is not supported"
    print(f"✅ Python version: {sys.version}")


def test_basic_math():
    """Test basic Python math operations."""
    assert 2 + 2 == 4
    assert 10 / 2 == 5.0
    assert 3 ** 2 == 9
    print("✅ Basic math operations work")


def test_basic_imports():
    """Test that basic Python modules can be imported."""
    import json
    import urllib
    import datetime
    
    assert json.dumps({"test": True}) == '{"test": true}'
    print("✅ Basic Python modules imported successfully")


def test_environment():
    """Test environment variables and working directory."""
    cwd = os.getcwd()
    pythonpath = os.environ.get('PYTHONPATH', '')
    
    print(f"Current working directory: {cwd}")
    print(f"PYTHONPATH: {pythonpath}")
    
    # This should always pass
    assert len(cwd) > 0
    print("✅ Environment check passed")


if __name__ == "__main__":
    # Run basic tests if called directly
    test_python_version()
    test_basic_math()
    test_basic_imports()
    test_environment()
    print("All basic tests passed!")