import logging
from typing import Annotated, Dict
import pandas as pd
from zenml import step
from zenml.client import Client

from src.model_comparator import ModelComparator

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker


@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def model_comparison_step(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame,
    y_train: pd.Series, 
    y_test: pd.Series
) -> Annotated[Dict[str, Dict], "model_comparison_results"]:
    """
    Compare multiple regression models and return performance metrics.
    
    Parameters:
    X_train, X_test: Training and testing features
    y_train, y_test: Training and testing targets
    
    Returns:
    Dictionary with model comparison results
    """
    logging.info("🔍 Starting model comparison step...")
    
    # Initialize model comparator
    comparator = ModelComparator()
    
    # Compare all models
    results = comparator.compare_models(X_train, X_test, y_train, y_test)
    
    # Print comparison table
    comparator.print_comparison_table()
    
    # Get best model info
    try:
        best_model_name, best_metrics = comparator.get_best_model()
        logging.info(f"🏆 Best performing model: {best_model_name}")
        logging.info(f"   R² Score: {best_metrics['R²']:.4f}")
        logging.info(f"   RMSE: ${best_metrics['RMSE']:,.2f}")
        
        # Add best model info to results
        results['_best_model'] = {
            'name': best_model_name,
            'metrics': best_metrics
        }
        
    except ValueError as e:
        logging.error(f"Could not determine best model: {e}")
    
    return results


if __name__ == "__main__":
    logging.info("Model comparison step module loaded successfully!")