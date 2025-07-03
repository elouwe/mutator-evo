# src/mutator_evo/operators/_uniform_crossover.py
import random
from typing import Dict, Any, Set
from .interfaces import IMutationOperator

class UniformCrossover(IMutationOperator):
    def apply(self, features: Dict[str, Any], config: Any, 
              feature_bank: Set[str], importance: Dict[str, float]) -> Dict[str, Any]:
        # Only apply if both parents have RL agents
        if "rl_agent" not in features:
            return features
            
        if not hasattr(config, 'top_strategies') or len(config.top_strategies) < 2:
            return features
            
        # Select a different parent from the top strategies
        other_parent = random.choice(config.top_strategies)
        if "rl_agent" not in other_parent.features:
            return features
            
        child_rl = {}
        for param in features["rl_agent"]:
            if param in other_parent.features["rl_agent"]:
                # Randomly select from either parent
                child_rl[param] = random.choice([
                    features["rl_agent"][param],
                    other_parent.features["rl_agent"][param]
                ])
            else:
                # Keep current value if not in other parent
                child_rl[param] = features["rl_agent"][param]
        
        return {**features, "rl_agent": child_rl}