import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataValidator:
    """
    A comprehensive data validation utility for ML pipelines.
    """
    
    def __init__(self):
        self.validation_report = {}
        self.warnings_list = []
        self.errors_list = []
    
    def validate_data_types(self, df: pd.DataFrame, expected_types: Dict[str, str]) -> bool:
        """
        Validate data types of DataFrame columns.
        
        Parameters:
        df: Input DataFrame
        expected_types: Dictionary mapping column names to expected types
        
        Returns:
        bool: True if all types match, False otherwise
        """
        logging.info("🔍 Validating data types...")
        type_issues = []
        
        for column, expected_type in expected_types.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                if expected_type not in actual_type:
                    type_issues.append(f"Column '{column}': expected {expected_type}, got {actual_type}")
            else:
                type_issues.append(f"Missing column: '{column}'")
        
        if type_issues:
            self.errors_list.extend(type_issues)
            logging.error(f"❌ Data type validation failed: {type_issues}")
            return False
        
        logging.info("✅ Data types validation passed")
        return True
    
    def check_missing_values(self, df: pd.DataFrame, max_missing_percent: float = 0.5) -> Dict:
        """
        Check for missing values in the dataset.
        
        Parameters:
        df: Input DataFrame
        max_missing_percent: Maximum allowed missing percentage per column
        
        Returns:
        Dict: Missing value report
        """
        logging.info("🔍 Checking for missing values...")
        
        missing_report = {}
        total_rows = len(df)
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percent = (missing_count / total_rows) * 100
            
            missing_report[column] = {
                'missing_count': missing_count,
                'missing_percent': missing_percent
            }
            
            if missing_percent > max_missing_percent * 100:
                warning_msg = f"Column '{column}' has {missing_percent:.1f}% missing values"
                self.warnings_list.append(warning_msg)
                logging.warning(f"⚠️ {warning_msg}")
        
        total_missing = sum([info['missing_count'] for info in missing_report.values()])
        logging.info(f"📊 Missing values summary: {total_missing} total missing values")
        
        return missing_report
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr', z_threshold: float = 3.0) -> Dict:
        """
        Detect outliers in numerical columns.
        
        Parameters:
        df: Input DataFrame
        method: 'iqr' or 'zscore'
        z_threshold: Z-score threshold for outlier detection
        
        Returns:
        Dict: Outlier detection report
        """
        logging.info(f"🔍 Detecting outliers using {method} method...")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outlier_report = {}
        
        for column in numerical_cols:
            outliers = []
            
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index.tolist()
            
            elif method == 'zscore':
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                outliers = df[z_scores > z_threshold].index.tolist()
            
            outlier_percent = (len(outliers) / len(df)) * 100
            outlier_report[column] = {
                'outlier_count': len(outliers),
                'outlier_percent': outlier_percent,
                'outlier_indices': outliers[:10]  # Store first 10 indices
            }
            
            if outlier_percent > 5:  # More than 5% outliers
                warning_msg = f"Column '{column}' has {outlier_percent:.1f}% outliers"
                self.warnings_list.append(warning_msg)
                logging.warning(f"⚠️ {warning_msg}")
        
        logging.info(f"📊 Outlier detection complete for {len(numerical_cols)} numerical columns")
        return outlier_report
    
    def check_data_drift(self, df_reference: pd.DataFrame, df_current: pd.DataFrame, 
                        threshold: float = 0.1) -> Dict:
        """
        Check for data drift between reference and current datasets.
        
        Parameters:
        df_reference: Reference dataset
        df_current: Current dataset
        threshold: Threshold for detecting significant drift
        
        Returns:
        Dict: Data drift report
        """
        logging.info("🔍 Checking for data drift...")
        
        drift_report = {}
        numerical_cols = df_reference.select_dtypes(include=[np.number]).columns
        
        for column in numerical_cols:
            if column in df_current.columns:
                ref_mean = df_reference[column].mean()
                curr_mean = df_current[column].mean()
                
                ref_std = df_reference[column].std()
                curr_std = df_current[column].std()
                
                mean_drift = abs(ref_mean - curr_mean) / (ref_std + 1e-8)
                std_drift = abs(ref_std - curr_std) / (ref_std + 1e-8)
                
                drift_report[column] = {
                    'mean_drift': mean_drift,
                    'std_drift': std_drift,
                    'significant_drift': mean_drift > threshold or std_drift > threshold
                }
                
                if drift_report[column]['significant_drift']:
                    warning_msg = f"Significant drift detected in '{column}'"
                    self.warnings_list.append(warning_msg)
                    logging.warning(f"⚠️ {warning_msg}")
        
        logging.info(f"📊 Data drift analysis complete for {len(numerical_cols)} columns")
        return drift_report
    
    def validate_value_ranges(self, df: pd.DataFrame, range_constraints: Dict[str, Tuple]) -> Dict:
        """
        Validate that column values are within expected ranges.
        
        Parameters:
        df: Input DataFrame
        range_constraints: Dict mapping column names to (min, max) tuples
        
        Returns:
        Dict: Range validation report
        """
        logging.info("🔍 Validating value ranges...")
        
        range_report = {}
        
        for column, (min_val, max_val) in range_constraints.items():
            if column in df.columns:
                out_of_range = df[(df[column] < min_val) | (df[column] > max_val)]
                violation_count = len(out_of_range)
                violation_percent = (violation_count / len(df)) * 100
                
                range_report[column] = {
                    'violations': violation_count,
                    'violation_percent': violation_percent,
                    'expected_range': (min_val, max_val),
                    'actual_range': (df[column].min(), df[column].max())
                }
                
                if violation_count > 0:
                    warning_msg = f"Column '{column}' has {violation_count} values outside range [{min_val}, {max_val}]"
                    self.warnings_list.append(warning_msg)
                    logging.warning(f"⚠️ {warning_msg}")
        
        logging.info(f"📊 Range validation complete for {len(range_constraints)} columns")
        return range_report
    
    def comprehensive_validation(self, df: pd.DataFrame, 
                               reference_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Run comprehensive data validation.
        
        Parameters:
        df: DataFrame to validate
        reference_df: Optional reference DataFrame for drift detection
        
        Returns:
        Dict: Complete validation report
        """
        logging.info("🚀 Starting comprehensive data validation...")
        
        # Reset validation state
        self.validation_report = {}
        self.warnings_list = []
        self.errors_list = []
        
        # Basic statistics
        self.validation_report['basic_stats'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Missing values check
        self.validation_report['missing_values'] = self.check_missing_values(df)
        
        # Outlier detection
        self.validation_report['outliers'] = self.detect_outliers(df)
        
        # Data drift (if reference provided)
        if reference_df is not None:
            self.validation_report['data_drift'] = self.check_data_drift(reference_df, df)
        
        # Range validation for common house price features
        if 'SalePrice' in df.columns:
            price_ranges = {
                'SalePrice': (10000, 1000000),  # Reasonable house price range
            }
            if 'Gr Liv Area' in df.columns:
                price_ranges['Gr Liv Area'] = (300, 10000)  # Living area range
            
            self.validation_report['range_validation'] = self.validate_value_ranges(df, price_ranges)
        
        # Summary
        self.validation_report['summary'] = {
            'total_warnings': len(self.warnings_list),
            'total_errors': len(self.errors_list),
            'warnings': self.warnings_list,
            'errors': self.errors_list,
            'validation_passed': len(self.errors_list) == 0
        }
        
        logging.info(f"🎯 Validation complete!")
        logging.info(f"   Warnings: {len(self.warnings_list)}")
        logging.info(f"   Errors: {len(self.errors_list)}")
        logging.info(f"   Status: {'✅ PASSED' if len(self.errors_list) == 0 else '❌ FAILED'}")
        
        return self.validation_report


if __name__ == "__main__":
    logging.info("Data Validator module loaded successfully!")