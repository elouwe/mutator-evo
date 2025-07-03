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
        
        # Собираем все возможные признаки
        all_features = set()
        for s in top_strats:
            all_features.update(s.features.keys())
        all_features = sorted(all_features)
        
        # Создаем матрицу признаков и целевую переменную
        X = []
        y = []
        for s in top_strats:
            row = []
            for feat in all_features:
                value = s.features.get(feat)
                
                # Обработка сложных признаков (например, rl_agent)
                if isinstance(value, dict):
                    # Для словарных признаков используем бинарный флаг
                    row.append(1)
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
        
        # Преобразуем в numpy массивы
        X = np.array(X)
        y = np.array(y)
        
        # Предобработка данных
        try:
            X_processed = self.preprocessor.fit_transform(X)
        except Exception as e:
            print(f"Data preprocessing error: {str(e)}")
            return {}
        
        # Обучаем модель
        try:
            self.model.fit(X_processed, y)
        except Exception as e:
            print(f"Model training error: {str(e)}")
            return {}
        
        # Вычисляем SHAP значения
        try:
            explainer = shap.Explainer(self.model)
            shap_values = explainer(X_processed)
            
            # Усредняем абсолютные значения SHAP для каждого признака
            feature_importances = np.abs(shap_values.values).mean(axis=0)
            
            # Создаем словарь важности признаков
            importance_dict = dict(zip(all_features, feature_importances))
            
            # Нормализуем важности от 0 до 1
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
    """Комбинирует SHAP и стандартный метод расчета важности"""
    def __init__(self, shap_weight=0.7):
        self.shap_weight = shap_weight
        self.default_calculator = DefaultFeatureImportanceCalculator()
        self.shap_calculator = ShapFeatureImportanceCalculator()
    
    def compute(self, top_strats: List) -> Dict[str, float]:
        # Стандартный метод
        default_importance = self.default_calculator.compute(top_strats)
        
        # SHAP метод
        shap_importance = self.shap_calculator.compute(top_strats)
        
        # Если SHAP не сработал, используем только стандартный метод
        if not shap_importance:
            return default_importance
        
        # Комбинируем результаты
        combined = {}
        all_features = set(default_importance.keys()) | set(shap_importance.keys())
        
        for feature in all_features:
            default_val = default_importance.get(feature, 0)
            shap_val = shap_importance.get(feature, 0)
            combined[feature] = (self.shap_weight * shap_val + 
                                (1 - self.shap_weight) * default_val)
        
        return combined