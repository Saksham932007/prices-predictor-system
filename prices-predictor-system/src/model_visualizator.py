import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModelVisualizator:
    """
    A utility class for creating visualizations of model performance and predictions.
    """
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_actual_vs_predicted(self, y_test: pd.Series, y_pred: np.ndarray, 
                                model_name: str = "Model") -> str:
        """
        Create actual vs predicted scatter plot.
        
        Returns:
        str: Path to saved plot
        """
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        plt.scatter(y_test, y_pred, alpha=0.6, s=50)
        
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, 
                label='Perfect Prediction')
        
        # Calculate R²
        r2 = np.corrcoef(y_test, y_pred)[0, 1] ** 2
        
        plt.xlabel('Actual House Prices ($)', fontsize=12)
        plt.ylabel('Predicted House Prices ($)', fontsize=12)
        plt.title(f'{model_name} - Actual vs Predicted Prices\nR² = {r2:.4f}', 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format axes
        plt.ticklabel_format(style='plain', axis='both')
        
        # Save plot
        filename = f"{self.output_dir}/actual_vs_predicted_{model_name.lower().replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved actual vs predicted plot: {filename}")
        return filename
    
    def plot_residuals(self, y_test: pd.Series, y_pred: np.ndarray, 
                      model_name: str = "Model") -> str:
        """
        Create residuals plot.
        
        Returns:
        str: Path to saved plot
        """
        residuals = y_test - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of residuals
        ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Residuals')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Residual Analysis', fontsize=16, fontweight='bold')
        
        # Save plot
        filename = f"{self.output_dir}/residuals_{model_name.lower().replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved residuals plot: {filename}")
        return filename
    
    def plot_model_comparison(self, comparison_results: Dict[str, Dict]) -> str:
        """
        Create model comparison bar chart.
        
        Parameters:
        comparison_results: Results from ModelComparator
        
        Returns:
        str: Path to saved plot
        """
        # Filter out error results and best model metadata
        valid_results = {name: metrics for name, metrics in comparison_results.items() 
                        if 'Error' not in metrics and name != '_best_model'}
        
        if not valid_results:
            logging.warning("No valid results to plot")
            return ""
        
        # Extract data for plotting
        models = list(valid_results.keys())
        r2_scores = [metrics['R²'] for metrics in valid_results.values()]
        rmse_scores = [metrics['RMSE'] for metrics in valid_results.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² comparison
        bars1 = ax1.bar(models, r2_scores, color='skyblue', alpha=0.7)
        ax1.set_ylabel('R² Score')
        ax1.set_title('Model Comparison - R² Score')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars1, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # RMSE comparison
        bars2 = ax2.bar(models, rmse_scores, color='lightcoral', alpha=0.7)
        ax2.set_ylabel('RMSE ($)')
        ax2.set_title('Model Comparison - RMSE')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars2, rmse_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_scores)*0.01,
                    f'${score:,.0f}', ha='center', va='bottom')
        
        # Rotate x-axis labels for better readability
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Save plot
        filename = f"{self.output_dir}/model_comparison.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved model comparison plot: {filename}")
        return filename
    
    def create_model_report(self, y_test: pd.Series, y_pred: np.ndarray,
                           model_name: str = "Model", 
                           comparison_results: Dict = None) -> List[str]:
        """
        Create a comprehensive visual report for model performance.
        
        Returns:
        List[str]: Paths to all generated plots
        """
        logging.info(f"📊 Creating visual report for {model_name}...")
        
        plots = []
        
        # Individual model plots
        plots.append(self.plot_actual_vs_predicted(y_test, y_pred, model_name))
        plots.append(self.plot_residuals(y_test, y_pred, model_name))
        
        # Model comparison plot if results provided
        if comparison_results:
            comparison_plot = self.plot_model_comparison(comparison_results)
            if comparison_plot:
                plots.append(comparison_plot)
        
        logging.info(f"✅ Created {len(plots)} visualization(s) in {self.output_dir}")
        return plots


if __name__ == "__main__":
    logging.info("Model Visualizator module loaded successfully!")