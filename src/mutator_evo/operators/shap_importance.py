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
                # Преобразуем булевы значения в числовые
                if isinstance(value, bool):
                    row.append(1 if value else 0)
                elif value is None:
                    row.append(0)  # Заполняем отсутствующие признаки нулями
                else:
                    row.append(float(value))
            X.append(row)
            y.append(s.score())
        
        # Преобразуем в numpy массивы
        X = np.array(X)
        y = np.array(y)
        
        # Предобработка данных
        X_processed = self.preprocessor.fit_transform(X)
        
        # Обучаем модель
        self.model.fit(X_processed, y)
        
        # Вычисляем SHAP значения
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