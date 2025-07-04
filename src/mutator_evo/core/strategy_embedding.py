# src/mutator_evo/core/strategy_embedding.py
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class StrategyEmbedding:
    name: str
    features: Dict[str, Any]
    oos_metrics: Optional[Dict[str, float]] = field(default=None)
    
    def __post_init__(self):
        self._cached_score = None
    
    @classmethod
    def create_random(cls, feature_bank: set) -> "StrategyEmbedding":
        # Create more diverse and aggressive strategies
        num_features = random.randint(4, min(8, len(feature_bank)))
        selected_features = random.sample(list(feature_bank), num_features)
        
        # Create meaningful parameter values
        features = {}
        for feat in selected_features:
            if 'period' in feat:
                features[feat] = random.randint(5, 25)
            elif 'trade_size' in feat:
                features[feat] = random.uniform(0.15, 0.4)
            elif 'stop_loss' in feat:
                features[feat] = random.uniform(0.01, 0.04)
            elif 'take_profit' in feat:
                features[feat] = random.uniform(0.03, 0.08)
            else:
                # 80% chance of true for boolean features
                features[feat] = random.random() > 0.2
        
        # 30% chance to include RL agent
        if random.random() < 0.3:
            features["rl_agent"] = {
                "hidden_layers": [random.randint(64, 256) for _ in range(random.randint(1, 3))],
                "learning_rate": random.uniform(1e-4, 1e-2),
                "gamma": random.uniform(0.9, 0.99),
                "epsilon": random.uniform(0.05, 0.3),
                "use_attention": random.random() > 0.7,
                "weights": None  
            }
                
        return cls(
            name=f"rand_{uuid.uuid4().hex[:8]}",
            features=features
        )
    
    def score(self) -> float:
        if self._cached_score is not None:
            return self._cached_score
            
        if self.oos_metrics is None:
            return 0.5
        else:
            # Clip Sharpe values to reasonable range
            sharpe = self.oos_metrics.get("oos_sharpe", 0)
            sharpe = max(min(sharpe, 10), -10)
            
            penalty = self.oos_metrics.get("overfitting_penalty", 0)
            
            # Include trade count in score
            trade_count = self.oos_metrics.get("trade_count", 0)
            trade_bonus = min(trade_count / 50, 2.0)
            
            # Safe calculation with constraints
            raw_score = sharpe - penalty + trade_bonus
            self._cached_score = max(-10.0, min(raw_score, 10.0))
        return self._cached_score
    
    def __repr__(self) -> str:
        return f"Strategy(name={self.name}, features={list(self.features.keys())}, score={self.score():.2f})"
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_cached_score'] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, '_cached_score'):
            self._cached_score = None
        if not hasattr(self, 'oos_metrics'):
            self.oos_metrics = None