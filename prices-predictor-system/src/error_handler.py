import logging
import traceback
import functools
from typing import Any, Callable, Optional, Dict
import pandas as pd
import numpy as np
from datetime import datetime


class MLPipelineError(Exception):
    """Base exception class for ML pipeline errors."""
    pass


class DataValidationError(MLPipelineError):
    """Exception raised for data validation errors."""
    pass


class ModelTrainingError(MLPipelineError):
    """Exception raised for model training errors."""
    pass


class FeatureEngineeringError(MLPipelineError):
    """Exception raised for feature engineering errors."""
    pass


class MLErrorHandler:
    """
    Comprehensive error handling utility for ML pipelines.
    """
    
    def __init__(self):
        self.error_log = []
        self.logger = logging.getLogger("mlpipeline")
    
    def log_error(self, error: Exception, context: str = "", additional_info: Dict = None):
        """
        Log error with detailed context information.
        """
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'traceback': traceback.format_exc(),
            'additional_info': additional_info or {}
        }
        
        self.error_log.append(error_entry)
        
        self.logger.error(f"❌ {context} - {type(error).__name__}: {str(error)}")
        if additional_info:
            self.logger.error(f"   Additional Info: {additional_info}")
    
    def safe_execute(self, func: Callable, *args, **kwargs) -> tuple:
        """
        Safely execute a function with comprehensive error handling.
        
        Returns:
        tuple: (success: bool, result: Any, error: Optional[Exception])
        """
        try:
            result = func(*args, **kwargs)
            return True, result, None
        except Exception as e:
            self.log_error(e, f"Function execution: {func.__name__}")
            return False, None, e
    
    def get_error_summary(self) -> Dict:
        """Get summary of all logged errors."""
        if not self.error_log:
            return {'total_errors': 0, 'error_types': {}}
        
        error_types = {}
        for error in self.error_log:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.error_log),
            'error_types': error_types,
            'recent_errors': self.error_log[-5:]  # Last 5 errors
        }


def handle_pipeline_errors(error_handler: Optional[MLErrorHandler] = None):
    """
    Decorator for handling pipeline step errors.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = error_handler or MLErrorHandler()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler.log_error(
                    e, 
                    f"Pipeline step: {func.__name__}",
                    {'args_count': len(args), 'kwargs': list(kwargs.keys())}
                )
                
                # Re-raise with more context
                raise MLPipelineError(f"Error in {func.__name__}: {str(e)}") from e
        
        return wrapper
    return decorator


def validate_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> pd.DataFrame:
    """
    Validate DataFrame with comprehensive checks.
    """
    logger = logging.getLogger("mlpipeline")
    
    if df is None:
        raise DataValidationError(f"{name} is None")
    
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"{name} is not a pandas DataFrame")
    
    if df.empty:
        raise DataValidationError(f"{name} is empty")
    
    if df.shape[0] == 0:
        raise DataValidationError(f"{name} has no rows")
    
    if df.shape[1] == 0:
        raise DataValidationError(f"{name} has no columns")
    
    # Check for all NaN columns
    all_nan_cols = df.columns[df.isnull().all()].tolist()
    if all_nan_cols:
        logger.warning(f"⚠️ {name} has columns with all NaN values: {all_nan_cols}")
    
    # Check for duplicate columns
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        raise DataValidationError(f"{name} has duplicate columns: {duplicate_cols}")
    
    logger.info(f"✅ {name} validation passed")
    return df


def validate_model_inputs(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                         y_train: pd.Series, y_test: pd.Series):
    """
    Validate model training inputs.
    """
    logger = logging.getLogger("mlpipeline")
    
    # Validate DataFrames
    validate_dataframe(X_train, "X_train")
    validate_dataframe(X_test, "X_test")
    
    # Validate Series
    if not isinstance(y_train, pd.Series):
        raise DataValidationError("y_train must be a pandas Series")
    
    if not isinstance(y_test, pd.Series):
        raise DataValidationError("y_test must be a pandas Series")
    
    # Check shapes
    if len(X_train) != len(y_train):
        raise DataValidationError(f"X_train and y_train length mismatch: {len(X_train)} vs {len(y_train)}")
    
    if len(X_test) != len(y_test):
        raise DataValidationError(f"X_test and y_test length mismatch: {len(X_test)} vs {len(y_test)}")
    
    # Check feature consistency
    if not X_train.columns.equals(X_test.columns):
        missing_in_test = set(X_train.columns) - set(X_test.columns)
        missing_in_train = set(X_test.columns) - set(X_train.columns)
        
        if missing_in_test:
            raise DataValidationError(f"Features missing in X_test: {missing_in_test}")
        if missing_in_train:
            raise DataValidationError(f"Features missing in X_train: {missing_in_train}")
    
    # Check for infinite values
    for df_name, df in [("X_train", X_train), ("X_test", X_test)]:
        inf_cols = df.columns[np.isinf(df.select_dtypes(include=[np.number])).any()].tolist()
        if inf_cols:
            raise DataValidationError(f"{df_name} contains infinite values in columns: {inf_cols}")
    
    # Check target values
    for y_name, y in [("y_train", y_train), ("y_test", y_test)]:
        if y.isnull().any():
            raise DataValidationError(f"{y_name} contains null values")
        
        if np.isinf(y).any():
            raise DataValidationError(f"{y_name} contains infinite values")
        
        if (y < 0).any():
            logger.warning(f"⚠️ {y_name} contains negative values")
    
    logger.info("✅ Model input validation passed")


def robust_prediction(model, X: pd.DataFrame, fallback_value: Optional[float] = None) -> np.ndarray:
    """
    Make robust predictions with error handling.
    """
    logger = logging.getLogger("mlpipeline")
    
    try:
        validate_dataframe(X, "Prediction input")
        predictions = model.predict(X)
        
        # Check for invalid predictions
        if np.isnan(predictions).any():
            logger.warning("⚠️ Model produced NaN predictions")
            if fallback_value is not None:
                predictions[np.isnan(predictions)] = fallback_value
        
        if np.isinf(predictions).any():
            logger.warning("⚠️ Model produced infinite predictions")
            if fallback_value is not None:
                predictions[np.isinf(predictions)] = fallback_value
        
        return predictions
    
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")
        if fallback_value is not None:
            logger.warning(f"🔄 Using fallback value: {fallback_value}")
            return np.full(len(X), fallback_value)
        raise ModelTrainingError(f"Prediction failed: {str(e)}") from e


# Global error handler instance
global_error_handler = MLErrorHandler()


if __name__ == "__main__":
    # Test error handling
    logger = logging.getLogger("mlpipeline")
    handler = MLErrorHandler()
    
    # Test safe execution
    def test_function(x, y):
        if y == 0:
            raise ValueError("Division by zero")
        return x / y
    
    success, result, error = handler.safe_execute(test_function, 10, 2)
    print(f"Success: {success}, Result: {result}")
    
    success, result, error = handler.safe_execute(test_function, 10, 0)
    print(f"Success: {success}, Error: {error}")
    
    print("Error summary:", handler.get_error_summary())