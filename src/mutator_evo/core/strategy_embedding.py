import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Callable

@dataclass
class StrategyEmbedding:
    name: str
    features: Dict[str, Any]
    _func: Callable = field(init=False, repr=False)
    
    def __post_init__(self):
        self._func = self._generate_func()
    
    @classmethod
    def create_random(cls, feature_bank: set) -> 'StrategyEmbedding':
        """Creates a random strategy from a bank of chips"""
        selected_features = {
            feat: random.uniform(0, 1) 
            for feat in random.sample(feature_bank, k=min(5, len(feature_bank)))
        }
        return cls(
            name=f"rand_{uuid.uuid4().hex[:8]}",
            features=selected_features
        )
    
    def _generate_func(self):
        """Generation of strategy function based on features"""
        def strategy_func(data):
            # Simple demo logic
            buy_condition = (
                self.features.get('use_ema', False) and 
                data.close > data.ema(20))
            sell_condition = (
                self.features.get('use_rsi', False) and 
                data.rsi > 70)
            return buy_condition and not sell_condition
        return strategy_func
    
    @property
    def func(self) -> Callable:
        return self._func
    
    def score(self) -> float:
        """Strategy evaluation (stub)"""
        base_score = random.uniform(-1, 2)
        # Complexity penalty
        complexity_penalty = len(self.features) * 0.01
        return max(-1, base_score - complexity_penalty)
    
    def estimate_score_without(self, feature: str) -> float:
        """Evaluation of the strategy without the specified feature"""
        return random.uniform(-1, 2)
    
    def __repr__(self) -> str:
        return f"Strategy(name={self.name}, features={len(self.features)})"
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_func']
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._func = self._generate_func()