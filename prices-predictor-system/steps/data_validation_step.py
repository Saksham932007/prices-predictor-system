import logging
from typing import Annotated, Dict
import pandas as pd
from zenml import step
from zenml.client import Client

from src.data_validator import DataValidator

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker


@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def data_validation_step(
    df: pd.DataFrame,
    reference_df: pd.DataFrame = None,
    validation_config: Dict = None
) -> Annotated[Dict, "validation_report"]:
    """
    Comprehensive data validation step for ML pipeline.
    
    Parameters:
    df: DataFrame to validate
    reference_df: Optional reference DataFrame for drift detection
    validation_config: Optional validation configuration
    
    Returns:
    Dict: Comprehensive validation report
    """
    logging.info("🔍 Starting data validation step...")
    
    # Initialize validator
    validator = DataValidator()
    
    # Run comprehensive validation
    validation_report = validator.comprehensive_validation(df, reference_df)
    
    # Log key metrics
    summary = validation_report['summary']
    logging.info(f"📊 Validation Summary:")
    logging.info(f"   Total rows: {validation_report['basic_stats']['total_rows']:,}")
    logging.info(f"   Total columns: {validation_report['basic_stats']['total_columns']}")
    logging.info(f"   Memory usage: {validation_report['basic_stats']['memory_usage_mb']:.2f} MB")
    logging.info(f"   Warnings: {summary['total_warnings']}")
    logging.info(f"   Errors: {summary['total_errors']}")
    
    # Check validation status
    if not summary['validation_passed']:
        logging.error("❌ Data validation failed!")
        for error in summary['errors']:
            logging.error(f"   ERROR: {error}")
        raise ValueError(f"Data validation failed with {summary['total_errors']} errors")
    
    if summary['total_warnings'] > 0:
        logging.warning(f"⚠️ Data validation passed with {summary['total_warnings']} warnings:")
        for warning in summary['warnings']:
            logging.warning(f"   WARNING: {warning}")
    else:
        logging.info("✅ Data validation passed with no warnings!")
    
    return validation_report


if __name__ == "__main__":
    logging.info("Data validation step module loaded successfully!")