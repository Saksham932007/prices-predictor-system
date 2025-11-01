import logging
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.base import RegressorMixin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModelComparator:
    """
    A class for comparing multiple regression models on the same dataset.
    """
    
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.results = {}
    
    def add_model(self, name: str, model: RegressorMixin):
        """Add a custom model to the comparison."""
        self.models[name] = model
    
    def compare_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      y_train: pd.Series, y_test: pd.Series) -> Dict[str, Dict]:
        """
        Compare all models and return performance metrics.
        
        Parameters:
        X_train, X_test: Training and testing features
        y_train, y_test: Training and testing targets
        
        Returns:
        Dictionary with model names as keys and metrics as values
        """
        logging.info("🔍 Starting model comparison...")
        
        for name, model in self.models.items():
            try:
                logging.info(f"Training {name}...")
                
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                          scoring='neg_mean_squared_error')
                cv_rmse = np.sqrt(-cv_scores.mean())
                
                self.results[name] = {
                    'RMSE': rmse,
                    'R²': r2,
                    'MSE': mse,
                    'CV_RMSE': cv_rmse
                }
                
                logging.info(f"✅ {name} - R²: {r2:.4f}, RMSE: ${rmse:,.2f}")
                
            except Exception as e:
                logging.error(f"❌ Error training {name}: {str(e)}")
                self.results[name] = {'Error': str(e)}
        
        return self.results
    
    def get_best_model(self) -> Tuple[str, Dict]:
        """Return the best performing model based on R² score."""
        if not self.results:
            raise ValueError("No models have been compared yet. Run compare_models() first.")
        
        valid_results = {name: metrics for name, metrics in self.results.items() 
                        if 'Error' not in metrics}
        
        if not valid_results:
            raise ValueError("No models trained successfully.")
        
        best_model_name = max(valid_results.keys(), 
                             key=lambda x: valid_results[x]['R²'])
        
        return best_model_name, valid_results[best_model_name]
    
    def print_comparison_table(self):
        """Print a formatted comparison table of all models."""
        if not self.results:
            print("No results to display. Run compare_models() first.")
            return
        
        print("\n" + "="*80)
        print("🏆 MODEL COMPARISON RESULTS")
        print("="*80)
        print(f"{'Model':<20} {'R²':<10} {'RMSE':<15} {'CV_RMSE':<15} {'Status'}")
        print("-"*80)
        
        for name, metrics in self.results.items():
            if 'Error' in metrics:
                print(f"{name:<20} {'N/A':<10} {'N/A':<15} {'N/A':<15} {'❌ Error'}")
            else:
                r2 = f"{metrics['R²']:.4f}"
                rmse = f"${metrics['RMSE']:,.0f}"
                cv_rmse = f"${metrics['CV_RMSE']:,.0f}"
                print(f"{name:<20} {r2:<10} {rmse:<15} {cv_rmse:<15} {'✅ Success'}")
        
        print("-"*80)
        
        try:
            best_name, best_metrics = self.get_best_model()
            print(f"🥇 Best Model: {best_name} (R² = {best_metrics['R²']:.4f})")
        except ValueError as e:
            print(f"❌ Could not determine best model: {e}")
        
        print("="*80)


if __name__ == "__main__":
    # Example usage
    logging.info("Model Comparator module loaded successfully!")