import yaml
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class ModelConfig:
    """Configuration for model training parameters."""
    model_type: str = "linear_regression"
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    
    # Linear Regression parameters
    fit_intercept: bool = True
    normalize: bool = False
    
    # Random Forest parameters
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2
    
    # XGBoost parameters
    xgb_learning_rate: float = 0.1
    xgb_max_depth: int = 6
    xgb_n_estimators: int = 100


@dataclass
class DataConfig:
    """Configuration for data processing parameters."""
    data_path: str = "extracted_data/AmesHousing.csv"
    target_column: str = "SalePrice"
    
    # Feature engineering
    enable_interaction_features: bool = True
    enable_polynomial_features: bool = True
    polynomial_degree: int = 2
    
    # Data validation
    max_missing_percent: float = 0.5
    outlier_method: str = "iqr"  # or "zscore"
    z_threshold: float = 3.0
    
    # Feature selection
    feature_selection_method: str = "variance_threshold"
    variance_threshold: float = 0.0


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    enable_caching: bool = False
    enable_visualization: bool = True
    enable_model_comparison: bool = True
    enable_data_validation: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/ml_pipeline.log"
    
    # Output
    output_dir: str = "results"
    save_predictions: bool = True
    save_models: bool = True


@dataclass
class MLConfig:
    """Master configuration class."""
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    pipeline: PipelineConfig = PipelineConfig()
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'MLConfig':
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            logging.warning(f"Config file {config_path} not found. Using default configuration.")
            return cls()
        
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MLConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        
        if 'pipeline' in config_dict:
            config.pipeline = PipelineConfig(**config_dict['pipeline'])
        
        return config
    
    def to_yaml(self, config_path: str):
        """Save configuration to YAML file."""
        config_dict = asdict(self)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False, indent=2)
        
        logging.info(f"Configuration saved to {config_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def update_from_env(self):
        """Update configuration from environment variables."""
        # Model configuration
        if os.getenv('MODEL_TYPE'):
            self.model.model_type = os.getenv('MODEL_TYPE')
        
        if os.getenv('RANDOM_STATE'):
            self.model.random_state = int(os.getenv('RANDOM_STATE'))
        
        if os.getenv('TEST_SIZE'):
            self.model.test_size = float(os.getenv('TEST_SIZE'))
        
        # Data configuration
        if os.getenv('DATA_PATH'):
            self.data.data_path = os.getenv('DATA_PATH')
        
        if os.getenv('TARGET_COLUMN'):
            self.data.target_column = os.getenv('TARGET_COLUMN')
        
        # Pipeline configuration
        if os.getenv('LOG_LEVEL'):
            self.pipeline.log_level = os.getenv('LOG_LEVEL')
        
        if os.getenv('OUTPUT_DIR'):
            self.pipeline.output_dir = os.getenv('OUTPUT_DIR')
        
        logging.info("Configuration updated from environment variables")
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        errors = []
        
        # Validate model config
        if self.model.test_size <= 0 or self.model.test_size >= 1:
            errors.append("test_size must be between 0 and 1")
        
        if self.model.cv_folds < 2:
            errors.append("cv_folds must be at least 2")
        
        # Validate data config
        if self.data.max_missing_percent < 0 or self.data.max_missing_percent > 1:
            errors.append("max_missing_percent must be between 0 and 1")
        
        if self.data.outlier_method not in ['iqr', 'zscore']:
            errors.append("outlier_method must be 'iqr' or 'zscore'")
        
        if self.data.polynomial_degree < 1:
            errors.append("polynomial_degree must be at least 1")
        
        # Validate pipeline config
        if self.pipeline.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            errors.append("log_level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
        
        if errors:
            for error in errors:
                logging.error(f"Configuration validation error: {error}")
            return False
        
        logging.info("✅ Configuration validation passed")
        return True


class ConfigManager:
    """Utility class for managing configurations."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config = None
    
    @property
    def config(self) -> MLConfig:
        """Get configuration, loading from file if not already loaded."""
        if self._config is None:
            self._config = MLConfig.from_yaml(self.config_path)
            self._config.update_from_env()
        return self._config
    
    def reload_config(self):
        """Reload configuration from file."""
        self._config = None
        logging.info(f"Configuration reloaded from {self.config_path}")
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        config_dict = self.config.to_dict()
        
        # Deep update
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(config_dict, updates)
        self._config = MLConfig.from_dict(config_dict)
        
        logging.info(f"Configuration updated: {updates}")
    
    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file."""
        save_path = path or self.config_path
        self.config.to_yaml(save_path)


# Global configuration manager
config_manager = ConfigManager()


if __name__ == "__main__":
    # Test configuration management
    config = MLConfig()
    
    # Test validation
    config.validate()
    
    # Test saving and loading
    config.to_yaml("test_config.yaml")
    loaded_config = MLConfig.from_yaml("test_config.yaml")
    
    print("Model config:", loaded_config.model)
    print("Data config:", loaded_config.data)
    print("Pipeline config:", loaded_config.pipeline)
    
    # Test config manager
    manager = ConfigManager("test_config.yaml")
    print("Manager config:", manager.config.model.model_type)
    
    # Clean up
    if os.path.exists("test_config.yaml"):
        os.remove("test_config.yaml")