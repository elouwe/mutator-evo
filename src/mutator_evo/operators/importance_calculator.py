# src/mutator_evo/operators/importance_calculator.py
from collections import Counter, defaultdict
from typing import List, Dict
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from mutator_evo.core.strategy_embedding import StrategyEmbedding

class DefaultFeatureImportanceCalculator:
    def compute(self, top_strats: List) -> Dict[str, float]:
        if not top_strats:
            return {}
            
        freq = Counter()
        for s in top_strats:
            for f in s.features:
                freq[f] += 1
        
        if not freq:
            return {}
            
        local_gain = defaultdict(float)

        for s in top_strats:
            base = s.score()
            for f in s.features:
                # Simplified gain estimation
                gain = base * 0.1  # Assume 10% impact per feature
                local_gain[f] += max(gain, 0)

        # Avoid division by zero
        max_lg = max(local_gain.values()) or 1
        max_fr = max(freq.values()) or 1

        return {
            f: (0.5 * (local_gain.get(f, 0) / max_lg) + (0.5 * (freq.get(f, 0) / max_fr)))
            for f in freq
        }

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
                
                # Специальная обработка для RL-агентов
                if feat == "rl_agent":
                    if isinstance(value, dict):
                        # Преобразуем конфиг RL в числовое представление
                        rl_value = 1
                        if "hidden_layers" in value:
                            rl_value += len(value["hidden_layers"]) * 0.1
                    else:
                        rl_value = 0
                    row.append(rl_value)
                # Обработка других признаков
                elif isinstance(value, bool):
                    row.append(1 if value else 0)
                elif value is None:
                    row.append(0)
                else:
                    try:
                        row.append(float(value))
                    except (TypeError, ValueError):
                        row.append(0)
            X.append(row)
            y.append(s.score())
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Preprocess data
        try:
            X_processed = self.preprocessor.fit_transform(X)
        except Exception as e:
            print(f"Data preprocessing error: {str(e)}")
            return {}
        
        # Train model
        try:
            self.model.fit(X_processed, y)
        except Exception as e:
            print(f"Model training error: {str(e)}")
            return {}
        
        # Calculate SHAP values
        try:
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
        except Exception as e:
            print(f"SHAP calculation error: {str(e)}")
            return {}

class HybridFeatureImportanceCalculator:
    """Combines SHAP and default importance calculation methods"""
    def __init__(self, shap_weight=0.7):
        self.shap_weight = shap_weight
        self.default_calculator = DefaultFeatureImportanceCalculator()
        self.shap_calculator = ShapFeatureImportanceCalculator()
    
    def compute(self, top_strats: List) -> Dict[str, float]:
        # Default method
        default_importance = self.default_calculator.compute(top_strats)
        
        # SHAP method
        shap_importance = self.shap_calculator.compute(top_strats)
        
        # If SHAP failed, use only default method
        if not shap_importance:
            return default_importance
        
        # Combine results
        combined = {}
        all_features = set(default_importance.keys()) | set(shap_importance.keys())
        
        for feature in all_features:
            default_val = default_importance.get(feature, 0)
            shap_val = shap_importance.get(feature, 0)
            combined[feature] = (self.shap_weight * shap_val + 
                                (1 - self.shap_weight) * default_val)
        
        return combined