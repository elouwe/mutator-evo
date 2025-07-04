# src/mutator_evo/operators/shap_importance.py
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from typing import List, Dict
from mutator_evo.core.strategy_embedding import StrategyEmbedding

class ShapFeatureImportanceCalculator:
    def __init__(self, n_estimators=50, max_depth=5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ])
    
    def compute(self, top_strats: List[StrategyEmbedding]) -> Dict[str, float]:
        """Calculate feature importance using SHAP values.
        
        Args:
            top_strats: List of top strategies to analyze
            
        Returns:
            Dictionary mapping feature names to their normalized importance scores (0-1)
        """
        if not top_strats or len(top_strats) < 5:
            return {}
        
        # Collect all possible features
        all_features = set()
        for s in top_strats:
            all_features.update(s.features.keys())
        all_features = sorted(all_features)
        
        # Create feature matrix and target variable
        X = []
        y = []
        for s in top_strats:
            row = []
            for feat in all_features:
                value = s.features.get(feat)
                # Convert boolean values to numeric
                if isinstance(value, bool):
                    row.append(1 if value else 0)
                elif value is None:
                    row.append(0)  # Fill missing features with zeros
                else:
                    row.append(float(value))
            X.append(row)
            y.append(s.score())
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Preprocess data
        X_processed = self.preprocessor.fit_transform(X)
        
        # Train model
        self.model.fit(X_processed, y)
        
        # Calculate SHAP values
        explainer = shap.Explainer(self.model)
        shap_values = explainer(X_processed)
        
        # Average absolute SHAP values for each feature
        feature_importances = np.abs(shap_values.values).mean(axis=0)
        
        # Create feature importance dictionary
        importance_dict = dict(zip(all_features, feature_importances))
        
        # Normalize importances to 0-1 range
        max_importance = max(importance_dict.values()) or 1
        normalized_importance = {
            k: v / max_importance 
            for k, v in importance_dict.items()
        }
        
        return normalized_importance