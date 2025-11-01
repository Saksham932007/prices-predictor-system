import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional


class MLLogger:
    """
    Enhanced logging utility for ML pipelines with structured logging.
    """
    
    def __init__(self):
        self.loggers = {}
    
    @staticmethod
    def setup_logging(
        name: str = "mlpipeline",
        level: str = "INFO",
        log_file: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        include_timestamp: bool = True
    ) -> logging.Logger:
        """
        Set up comprehensive logging configuration.
        
        Parameters:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
        include_timestamp: Whether to include timestamp in log messages
        
        Returns:
        Configured logger instance
        """
        logger = logging.getLogger(name)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        logger.setLevel(getattr(logging, level.upper()))
        
        # Create formatter
        if include_timestamp:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            formatter = logging.Formatter(
                '%(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # Rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def log_function_call(func):
        """
        Decorator to log function calls with parameters and execution time.
        """
        def wrapper(*args, **kwargs):
            logger = logging.getLogger("mlpipeline")
            func_name = func.__name__
            
            # Log function start
            logger.info(f"🚀 Starting {func_name}")
            if args or kwargs:
                logger.debug(f"   Parameters: args={len(args)}, kwargs={list(kwargs.keys())}")
            
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Completed {func_name} in {execution_time:.2f}s")
                return result
            
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"❌ Failed {func_name} after {execution_time:.2f}s: {str(e)}")
                raise
        
        return wrapper
    
    @staticmethod
    def log_data_info(df, name: str = "DataFrame"):
        """
        Log detailed information about a DataFrame.
        """
        logger = logging.getLogger("mlpipeline")
        logger.info(f"📊 {name} Info:")
        logger.info(f"   Shape: {df.shape}")
        logger.info(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        logger.info(f"   Columns: {list(df.columns)}")
        
        # Missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.info(f"   Missing values: {missing[missing > 0].to_dict()}")
        else:
            logger.info("   Missing values: None")
    
    @staticmethod
    def log_model_metrics(metrics: dict, model_name: str = "Model"):
        """
        Log model performance metrics in a structured format.
        """
        logger = logging.getLogger("mlpipeline")
        logger.info(f"🎯 {model_name} Performance Metrics:")
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                if 'error' in metric.lower() or 'mse' in metric.lower():
                    logger.info(f"   {metric}: {value:,.2f}")
                elif 'r2' in metric.lower() or 'score' in metric.lower():
                    logger.info(f"   {metric}: {value:.4f}")
                else:
                    logger.info(f"   {metric}: {value:.2f}")
            else:
                logger.info(f"   {metric}: {value}")
    
    @staticmethod
    def log_pipeline_stage(stage_name: str, status: str = "START"):
        """
        Log pipeline stage transitions.
        """
        logger = logging.getLogger("mlpipeline")
        
        if status == "START":
            logger.info(f"🔄 PIPELINE STAGE: {stage_name} - STARTED")
        elif status == "COMPLETE":
            logger.info(f"✅ PIPELINE STAGE: {stage_name} - COMPLETED")
        elif status == "ERROR":
            logger.error(f"❌ PIPELINE STAGE: {stage_name} - FAILED")
        else:
            logger.info(f"ℹ️ PIPELINE STAGE: {stage_name} - {status}")


# Global logger instance
ml_logger = MLLogger()

# Configure default logger
default_logger = ml_logger.setup_logging(
    name="mlpipeline",
    level="INFO",
    log_file="logs/ml_pipeline.log",
    include_timestamp=True
)


if __name__ == "__main__":
    # Test logging setup
    logger = MLLogger.setup_logging(
        name="test_logger",
        level="DEBUG",
        log_file="logs/test.log"
    )
    
    logger.info("Test logging configuration successful!")
    logger.debug("Debug message test")
    logger.warning("Warning message test")
    
    # Test data logging
    import pandas as pd
    test_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    MLLogger.log_data_info(test_df, "Test DataFrame")
    
    # Test metrics logging
    test_metrics = {'R²': 0.85, 'RMSE': 15000.50, 'MAE': 12000.25}
    MLLogger.log_model_metrics(test_metrics, "Test Model")