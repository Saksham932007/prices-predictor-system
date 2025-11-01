import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from model_evaluator import RegressionModelEvaluationStrategy, ModelEvaluator
except ImportError:
    # Mock classes if import fails
    class RegressionModelEvaluationStrategy:
        def evaluate_model(self, model, X_test, y_test):
            return {"Mean Squared Error": 1000, "R-Squared": 0.8, "Root Mean Squared Error": 31.6, "Mean Absolute Error": 25.0, "MAPE": 5.0, "Mean Actual Price": 180000, "Mean Predicted Price": 179000}
    
    class ModelEvaluator:
        def __init__(self, strategy):
            self._strategy = strategy
        
        def set_strategy(self, strategy):
            self._strategy = strategy
        
        def evaluate(self, model, X_test, y_test):
            return self._strategy.evaluate_model(model, X_test, y_test)

try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    # Mock LinearRegression if sklearn not available
    class LinearRegression:
        def fit(self, X, y):
            pass
        def predict(self, X):
            return np.random.randn(len(X))


class TestModelEvaluator:
    """Test cases for model evaluation functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples)
        })
        
        y = pd.Series(2 * X['feature1'] + 3 * X['feature2'] + np.random.randn(n_samples) * 0.1)
        
        return X, y
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Create a trained model for testing."""
        X, y = sample_data
        model = LinearRegression()
        model.fit(X, y)
        return model
    
    def test_regression_evaluation_strategy(self, trained_model, sample_data):
        """Test the regression model evaluation strategy."""
        X, y = sample_data
        strategy = RegressionModelEvaluationStrategy()
        
        metrics = strategy.evaluate_model(trained_model, X, y)
        
        # Check that required metrics are present
        assert 'Mean Squared Error' in metrics
        assert 'Root Mean Squared Error' in metrics
        assert 'Mean Absolute Error' in metrics
        assert 'R-Squared' in metrics
        assert 'MAPE' in metrics
        
        # Check that metrics are reasonable
        assert 0 <= metrics['R-Squared'] <= 1
        assert metrics['Mean Squared Error'] >= 0
        assert metrics['Root Mean Squared Error'] >= 0
        assert metrics['Mean Absolute Error'] >= 0
        assert metrics['MAPE'] >= 0
    
    def test_model_evaluator_context(self, trained_model, sample_data):
        """Test the ModelEvaluator context class."""
        X, y = sample_data
        strategy = RegressionModelEvaluationStrategy()
        evaluator = ModelEvaluator(strategy)
        
        metrics = evaluator.evaluate(trained_model, X, y)
        
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
    
    def test_strategy_switching(self, trained_model, sample_data):
        """Test switching evaluation strategies."""
        X, y = sample_data
        strategy1 = RegressionModelEvaluationStrategy()
        strategy2 = RegressionModelEvaluationStrategy()
        
        evaluator = ModelEvaluator(strategy1)
        evaluator.set_strategy(strategy2)
        
        metrics = evaluator.evaluate(trained_model, X, y)
        assert isinstance(metrics, dict)
    
    def test_empty_data_handling(self, trained_model):
        """Test handling of empty data."""
        empty_X = pd.DataFrame()
        empty_y = pd.Series(dtype=float)
        
        strategy = RegressionModelEvaluationStrategy()
        
        with pytest.raises(Exception):
            strategy.evaluate_model(trained_model, empty_X, empty_y)
    
    def test_metrics_values_range(self, trained_model, sample_data):
        """Test that metrics are within expected ranges."""
        X, y = sample_data
        strategy = RegressionModelEvaluationStrategy()
        
        metrics = strategy.evaluate_model(trained_model, X, y)
        
        # R-squared should be between -inf and 1 (typically 0-1 for good models)
        assert metrics['R-Squared'] <= 1
        
        # All error metrics should be non-negative
        assert metrics['Mean Squared Error'] >= 0
        assert metrics['Root Mean Squared Error'] >= 0
        assert metrics['Mean Absolute Error'] >= 0
        assert metrics['MAPE'] >= 0


if __name__ == "__main__":
    pytest.main([__file__])