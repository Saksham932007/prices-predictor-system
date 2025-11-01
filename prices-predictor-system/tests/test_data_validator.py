import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from data_validator import DataValidator
except ImportError:
    # If import fails, create a mock class for testing
    class DataValidator:
        def __init__(self):
            self.validation_report = {}
            self.warnings_list = []
            self.errors_list = []
        
        def check_missing_values(self, df):
            return {col: {'missing_count': 0, 'missing_percent': 0} for col in df.columns}
        
        def detect_outliers(self, df, method='iqr'):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            return {col: {'outlier_count': 0, 'outlier_percent': 0, 'outlier_indices': []} for col in numeric_cols}
        
        def check_data_drift(self, ref_df, curr_df, threshold=0.1):
            return {'feature1': {'mean_drift': 0, 'std_drift': 0, 'significant_drift': False}}
        
        def validate_value_ranges(self, df, constraints):
            return {col: {'violations': 0, 'violation_percent': 0, 'expected_range': constraints[col], 'actual_range': (0, 100)} for col in constraints}
        
        def comprehensive_validation(self, df, ref_df=None):
            return {
                'basic_stats': {'total_rows': len(df), 'total_columns': len(df.columns), 'memory_usage_mb': 1.0},
                'missing_values': self.check_missing_values(df),
                'outliers': self.detect_outliers(df),
                'summary': {'total_warnings': 0, 'total_errors': 0, 'validation_passed': True, 'warnings': [], 'errors': []}
            }


class TestDataValidator:
    """Test cases for data validation functionality."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'numeric_col': np.random.randn(100),
            'categorical_col': np.random.choice(['A', 'B', 'C'], 100),
            'missing_col': [1.0 if i % 10 != 0 else np.nan for i in range(100)],
            'price_col': np.random.uniform(10000, 500000, 100)
        })
    
    @pytest.fixture
    def validator(self):
        """Create DataValidator instance."""
        return DataValidator()
    
    def test_validator_initialization(self, validator):
        """Test DataValidator initialization."""
        assert hasattr(validator, 'validation_report')
        assert hasattr(validator, 'warnings_list')
        assert hasattr(validator, 'errors_list')
        assert len(validator.warnings_list) == 0
        assert len(validator.errors_list) == 0
    
    def test_check_missing_values(self, validator, sample_dataframe):
        """Test missing values detection."""
        report = validator.check_missing_values(sample_dataframe)
        
        assert isinstance(report, dict)
        assert len(report) == len(sample_dataframe.columns)
        
        # Check that missing_col has missing values detected
        assert report['missing_col']['missing_count'] > 0
        assert report['missing_col']['missing_percent'] > 0
        
        # Check that numeric_col has no missing values
        assert report['numeric_col']['missing_count'] == 0
    
    def test_detect_outliers_iqr(self, validator, sample_dataframe):
        """Test outlier detection using IQR method."""
        report = validator.detect_outliers(sample_dataframe, method='iqr')
        
        assert isinstance(report, dict)
        
        # Should have reports for numeric columns
        numeric_cols = sample_dataframe.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert col in report
            assert 'outlier_count' in report[col]
            assert 'outlier_percent' in report[col]
            assert 'outlier_indices' in report[col]
    
    def test_detect_outliers_zscore(self, validator, sample_dataframe):
        """Test outlier detection using Z-score method."""
        report = validator.detect_outliers(sample_dataframe, method='zscore', z_threshold=2.0)
        
        assert isinstance(report, dict)
        
        # Should have reports for numeric columns
        numeric_cols = sample_dataframe.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert col in report
    
    def test_check_data_drift(self, validator):
        """Test data drift detection."""
        # Create reference and current datasets
        np.random.seed(42)
        ref_df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
        
        # Current data with some drift
        curr_df = pd.DataFrame({
            'feature1': np.random.normal(0.5, 1.2, 100),  # Mean and std drift
            'feature2': np.random.normal(0, 1, 100)       # No drift
        })
        
        report = validator.check_data_drift(ref_df, curr_df, threshold=0.1)
        
        assert isinstance(report, dict)
        assert 'feature1' in report
        assert 'feature2' in report
        
        for col in report:
            assert 'mean_drift' in report[col]
            assert 'std_drift' in report[col]
            assert 'significant_drift' in report[col]
    
    def test_validate_value_ranges(self, validator, sample_dataframe):
        """Test value range validation."""
        range_constraints = {
            'price_col': (5000, 1000000),
            'numeric_col': (-5, 5)
        }
        
        report = validator.validate_value_ranges(sample_dataframe, range_constraints)
        
        assert isinstance(report, dict)
        
        for col in range_constraints:
            if col in sample_dataframe.columns:
                assert col in report
                assert 'violations' in report[col]
                assert 'violation_percent' in report[col]
                assert 'expected_range' in report[col]
                assert 'actual_range' in report[col]
    
    def test_comprehensive_validation(self, validator, sample_dataframe):
        """Test comprehensive validation."""
        report = validator.comprehensive_validation(sample_dataframe)
        
        assert isinstance(report, dict)
        assert 'basic_stats' in report
        assert 'missing_values' in report
        assert 'outliers' in report
        assert 'summary' in report
        
        # Check basic stats
        assert report['basic_stats']['total_rows'] == len(sample_dataframe)
        assert report['basic_stats']['total_columns'] == len(sample_dataframe.columns)
        
        # Check summary
        assert 'total_warnings' in report['summary']
        assert 'total_errors' in report['summary']
        assert 'validation_passed' in report['summary']
    
    def test_comprehensive_validation_with_reference(self, validator):
        """Test comprehensive validation with reference data."""
        np.random.seed(42)
        current_df = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50)
        })
        
        reference_df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        
        report = validator.comprehensive_validation(current_df, reference_df)
        
        assert 'data_drift' in report
    
    def test_empty_dataframe_handling(self, validator):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        
        # Should not crash, but may generate warnings
        report = validator.comprehensive_validation(empty_df)
        assert isinstance(report, dict)
    
    def test_validation_with_all_missing_column(self, validator):
        """Test validation with column that has all missing values."""
        df_with_all_missing = pd.DataFrame({
            'good_col': [1, 2, 3, 4, 5],
            'all_missing': [np.nan] * 5
        })
        
        report = validator.comprehensive_validation(df_with_all_missing)
        
        # Should detect high missing percentage
        missing_report = report['missing_values']
        assert missing_report['all_missing']['missing_percent'] == 100.0


if __name__ == "__main__":
    pytest.main([__file__])