# src/mutator_evo/operators/rl_mutation.py
import random
import numpy as np
from typing import Dict, Any, Set
from .interfaces import IMutationOperator

class RLMutation(IMutationOperator):
    def apply(self, features: Dict[str, Any], config: Any, 
              feature_bank: Set[str], importance: Dict[str, float]) -> Dict[str, Any]:
        if "rl_agent" not in features:
            return features
            
        mutated = features.copy()
        rl_params = mutated["rl_agent"]
        
        if "hidden_layers" in rl_params:
            layers = rl_params["hidden_layers"]
            # --- Limit the number of layers to 3 ---
            if len(layers) > 3:
                layers = layers[:3]
                rl_params["hidden_layers"] = layers
                
            if layers and random.random() < 0.3:
                layer_idx = random.randint(0, len(layers)-1)
                layers[layer_idx] = max(8, layers[layer_idx] + random.randint(-4, 4))
            elif random.random() < 0.1:
                if len(layers) < 3 and random.random() > 0.5:
                    layers.append(random.randint(32, 128))
                elif len(layers) > 1:
                    layers.pop(random.randint(0, len(layers)-1))
        
        for param in ["learning_rate", "gamma", "epsilon"]:
            if param in rl_params:
                rl_params[param] *= random.uniform(0.8, 1.2)
                
        if "weights" in rl_params and random.random() < 0.4:
            new_weights = []
            for weight_array in rl_params["weights"]:
                noise = np.random.normal(0, 0.1, weight_array.shape)
                new_weights.append(weight_array + noise)
            rl_params["weights"] = new_weights
            
        return mutated