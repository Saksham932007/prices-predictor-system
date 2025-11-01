import logging
import pandas as pd
import numpy as np
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class EnhancedFeatureEngineer:
    """
    Enhanced feature engineering class with interaction features and domain-specific transformations.
    """
    
    def __init__(self):
        self.feature_history = []
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features based on domain knowledge.
        
        Parameters:
        df: Input DataFrame
        
        Returns:
        DataFrame with new interaction features
        """
        logging.info("🔧 Creating interaction features...")
        df_enhanced = df.copy()
        
        # Check if required columns exist
        required_cols = ['Overall Qual', 'Gr Liv Area', 'Total Bsmt SF', 'Garage Area']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if 'Overall Qual' in df.columns and 'Gr Liv Area' in df.columns:
            # Quality * Living Area interaction
            df_enhanced['Quality_Area_Interaction'] = df['Overall Qual'] * df['Gr Liv Area']
            logging.info("✅ Added Quality_Area_Interaction feature")
        
        if 'Gr Liv Area' in df.columns:
            # Total area features
            total_area_cols = [col for col in ['Total Bsmt SF', 'Garage Area'] if col in df.columns]
            if total_area_cols:
                df_enhanced['Total_Area'] = df['Gr Liv Area'] + df[total_area_cols].sum(axis=1)
                logging.info("✅ Added Total_Area feature")
        
        # Price per square foot (if SalePrice available - for training data)
        if 'SalePrice' in df.columns and 'Gr Liv Area' in df.columns:
            df_enhanced['Price_Per_SqFt'] = df['SalePrice'] / (df['Gr Liv Area'] + 1)  # +1 to avoid division by zero
            logging.info("✅ Added Price_Per_SqFt feature")
        
        # Age-related features
        if 'Year Built' in df.columns:
            current_year = 2023  # Use a reference year
            df_enhanced['House_Age'] = current_year - df['Year Built']
            logging.info("✅ Added House_Age feature")
        
        if 'Year Remod/Add' in df.columns:
            df_enhanced['Years_Since_Remodel'] = current_year - df['Year Remod/Add']
            logging.info("✅ Added Years_Since_Remodel feature")
        
        # Bathroom ratio features
        bathroom_cols = [col for col in df.columns if 'Bath' in col]
        if len(bathroom_cols) >= 2:
            df_enhanced['Total_Bathrooms'] = df[bathroom_cols].sum(axis=1)
            logging.info("✅ Added Total_Bathrooms feature")
        
        # Room ratio features
        if 'TotRms AbvGrd' in df.columns and 'Gr Liv Area' in df.columns:
            df_enhanced['Room_Size_Ratio'] = df['Gr Liv Area'] / (df['TotRms AbvGrd'] + 1)
            logging.info("✅ Added Room_Size_Ratio feature")
        
        self.feature_history.append("interaction_features")
        logging.info(f"🎯 Created {len(df_enhanced.columns) - len(df.columns)} new interaction features")
        
        return df_enhanced
    
    def create_polynomial_features(self, df: pd.DataFrame, features: List[str], degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features for specified columns.
        
        Parameters:
        df: Input DataFrame
        features: List of feature names to create polynomials for
        degree: Polynomial degree (default: 2)
        
        Returns:
        DataFrame with polynomial features
        """
        logging.info(f"🔧 Creating degree-{degree} polynomial features for: {features}")
        df_poly = df.copy()
        
        for feature in features:
            if feature in df.columns:
                for d in range(2, degree + 1):
                    new_feature_name = f"{feature}_poly_{d}"
                    df_poly[new_feature_name] = df[feature] ** d
                    logging.info(f"✅ Added {new_feature_name}")
        
        self.feature_history.append(f"polynomial_features_degree_{degree}")
        return df_poly
    
    def create_binned_features(self, df: pd.DataFrame, features: Dict[str, int]) -> pd.DataFrame:
        """
        Create binned (discretized) features.
        
        Parameters:
        df: Input DataFrame
        features: Dictionary with feature names as keys and number of bins as values
        
        Returns:
        DataFrame with binned features
        """
        logging.info(f"🔧 Creating binned features for: {list(features.keys())}")
        df_binned = df.copy()
        
        for feature, n_bins in features.items():
            if feature in df.columns:
                new_feature_name = f"{feature}_binned"
                df_binned[new_feature_name] = pd.cut(df[feature], bins=n_bins, labels=False)
                logging.info(f"✅ Added {new_feature_name} with {n_bins} bins")
        
        self.feature_history.append("binned_features")
        return df_binned
    
    def create_log_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Create log-transformed features for skewed variables.
        
        Parameters:
        df: Input DataFrame
        features: List of feature names to log-transform
        
        Returns:
        DataFrame with log-transformed features
        """
        logging.info(f"🔧 Creating log-transformed features for: {features}")
        df_log = df.copy()
        
        for feature in features:
            if feature in df.columns:
                # Add 1 to handle zero values
                new_feature_name = f"{feature}_log"
                df_log[new_feature_name] = np.log1p(df[feature])
                logging.info(f"✅ Added {new_feature_name}")
        
        self.feature_history.append("log_features")
        return df_log
    
    def apply_all_enhancements(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive feature engineering pipeline.
        
        Parameters:
        df: Input DataFrame
        
        Returns:
        Enhanced DataFrame with all transformations
        """
        logging.info("🚀 Starting comprehensive feature engineering...")
        
        # Start with original data
        df_enhanced = df.copy()
        
        # 1. Create interaction features
        df_enhanced = self.create_interaction_features(df_enhanced)
        
        # 2. Create polynomial features for key numerical columns
        numerical_cols = df_enhanced.select_dtypes(include=[np.number]).columns.tolist()
        key_features = [col for col in ['Gr Liv Area', 'Overall Qual', 'Total Bsmt SF'] 
                       if col in numerical_cols]
        if key_features:
            df_enhanced = self.create_polynomial_features(df_enhanced, key_features[:2], degree=2)
        
        # 3. Create log features for skewed variables
        skewed_features = [col for col in ['Gr Liv Area', 'Total Bsmt SF', 'Garage Area'] 
                          if col in numerical_cols]
        if skewed_features:
            df_enhanced = self.create_log_features(df_enhanced, skewed_features[:2])
        
        # 4. Create binned features for some continuous variables
        if 'Overall Qual' in df_enhanced.columns:
            df_enhanced = self.create_binned_features(df_enhanced, {'Overall Qual': 5})
        
        logging.info(f"🎉 Feature engineering complete!")
        logging.info(f"   Original features: {len(df.columns)}")
        logging.info(f"   Enhanced features: {len(df_enhanced.columns)}")
        logging.info(f"   New features added: {len(df_enhanced.columns) - len(df.columns)}")
        
        return df_enhanced
    
    def get_feature_importance_summary(self, df_original: pd.DataFrame, df_enhanced: pd.DataFrame) -> Dict:
        """
        Get summary of feature engineering transformations.
        
        Returns:
        Dictionary with transformation summary
        """
        summary = {
            'original_features': len(df_original.columns),
            'enhanced_features': len(df_enhanced.columns),
            'new_features': len(df_enhanced.columns) - len(df_original.columns),
            'transformations_applied': self.feature_history.copy(),
            'new_feature_names': [col for col in df_enhanced.columns if col not in df_original.columns]
        }
        
        return summary


if __name__ == "__main__":
    logging.info("Enhanced Feature Engineer module loaded successfully!")