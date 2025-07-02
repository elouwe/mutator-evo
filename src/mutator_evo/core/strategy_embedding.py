# src/mutator_evo/core/strategy_embedding.py
import random
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Callable

@dataclass
class StrategyEmbedding:
    name: str
    features: Dict[str, Any]
    
    def __post_init__(self):
        self._func = self._generate_func()
    
    @classmethod
    def create_random(cls, feature_bank: set) -> "StrategyEmbedding":
        """Create random strategy from feature bank"""
        num_features = min(5, len(feature_bank))
        selected_features = random.sample(list(feature_bank), num_features)
        return cls(
            name=f"rand_{uuid.uuid4().hex[:8]}",
            features={feat: random.uniform(0, 1) for feat in selected_features}
        )
    
    def _generate_func(self):
        """Generate strategy function based on features"""
        def strategy_func(data):
            # Example strategy logic
            if self.features.get('use_ema', False) and data.close > data.ema(20):
                return True  # Buy signal
            return False
        return strategy_func
    
    @property
    def func(self) -> Callable:
        return self._func
    
    def score(self) -> float:
        """Strategy score (stub implementation)"""
        # More realistic dummy score
        base_score = sum(hash(f) % 100 / 100 for f in self.features) / len(self.features)
        complexity_penalty = len(self.features) * 0.01
        return max(0, base_score - complexity_penalty)
    
    def estimate_score_without(self, feature: str) -> float:
        """Estimate score without a specific feature"""
        if feature not in self.features:
            return self.score()
            
        features_without = {k: v for k, v in self.features.items() if k != feature}
        if not features_without:
            return 0
            
        base_score = sum(hash(f) % 100 / 100 for f in features_without) / len(features_without)
        complexity_penalty = (len(features_without)) * 0.01
        return max(0, base_score - complexity_penalty)
    
    def __repr__(self) -> str:
        return f"Strategy(name={self.name}, features={len(self.features)})"
    
    # Serialization methods
    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't serialize function
        if '_func' in state:
            del state['_func']
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Regenerate function after deserialization
        self._func = self._generate_func()