import logging
from typing import Annotated, List, Dict
import pandas as pd
import numpy as np
from zenml import step
from zenml.client import Client

from src.model_visualizator import ModelVisualizator

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker


@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def visualization_step(
    y_test: pd.Series,
    model,  # Trained model
    X_test: pd.DataFrame,
    model_name: str = "Model",
    comparison_results: Dict = None
) -> Annotated[List[str], "visualization_paths"]:
    """
    Create comprehensive visualizations for model performance.
    
    Parameters:
    y_test: True target values
    model: Trained model for predictions
    X_test: Test features
    model_name: Name of the model for plot titles
    comparison_results: Optional model comparison results
    
    Returns:
    List of paths to generated visualization files
    """
    logging.info(f"📊 Creating visualizations for {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Initialize visualizator
    viz = ModelVisualizator()
    
    # Create comprehensive report
    plot_paths = viz.create_model_report(
        y_test=y_test,
        y_pred=y_pred,
        model_name=model_name,
        comparison_results=comparison_results
    )
    
    logging.info(f"✅ Generated {len(plot_paths)} visualization(s)")
    
    return plot_paths


if __name__ == "__main__":
    logging.info("Visualization step module loaded successfully!")