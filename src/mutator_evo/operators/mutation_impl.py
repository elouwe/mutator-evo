# src/mutator_evo/operators/mutation_impl.py
import random
from typing import Dict, Any, Set

class DropMutation:
    def apply(self, features: Dict[str, Any], config: Any, 
              feature_bank: Set[str], importance: Dict[str, float]) -> Dict[str, Any]:
        # Create a copy to avoid modifying the original dict
        result = dict(features)
        
        if random.random() < config.mutation_probs.get("drop", 0) and len(result) > 1:
            # Remove least important features first
            if importance:
                # Filter only existing features that have importance scores
                existing_features = [k for k in result.keys() if k in importance]
                if existing_features:
                    to_drop = min(existing_features, key=lambda k: importance[k])
                else:
                    to_drop = random.choice(list(result.keys()))
            else:
                to_drop = random.choice(list(result.keys()))
            result.pop(to_drop)
        return result

class AddMutation:
    def apply(self, features: Dict[str, Any], config: Any, 
              feature_bank: Set[str], importance: Dict[str, float]) -> Dict[str, Any]:
        result = dict(features)
        
        if random.random() < config.mutation_probs.get("add", 0) and feature_bank:
            available = feature_bank - set(result.keys())
            if available:
                # Prioritize adding more important features
                if importance:
                    # Get top 3 most important available features
                    sorted_available = sorted(
                        available, 
                        key=lambda k: importance.get(k, 0), 
                        reverse=True
                    )[:3]
                    # Choose randomly from top important features
                    to_add = random.choice(sorted_available) if sorted_available else random.choice(list(available))
                else:
                    to_add = random.choice(list(available))
                result[to_add] = random.random()  # Assign random initial value
        return result

class ShiftMutation:
    def apply(self, features: Dict[str, Any], config: Any, 
              feature_bank: Set[str], importance: Dict[str, float]) -> Dict[str, Any]:
        result = dict(features)
        
        for key in list(result.keys()):
            # Only modify numeric features (skip booleans)
            if isinstance(result[key], (int, float)) and not isinstance(result[key], bool):
                if random.random() < config.mutation_probs.get("shift", 0):
                    # Apply random shift between 0.8x and 1.2x original value
                    shift_factor = 0.8 + 0.4 * random.random()
                    result[key] *= shift_factor
        return result

class InvertMutation:
    def apply(self, features: Dict[str, Any], config: Any, 
              feature_bank: Set[str], importance: Dict[str, float]) -> Dict[str, Any]:
        # Create a copy to avoid modifying the original dict
        result = dict(features)
        
        for key in list(result.keys()):
            # Only invert boolean values
            if isinstance(result[key], bool) and random.random() < config.mutation_probs.get("invert", 0):
                result[key] = not result[key]
        return result

class MetabBoostMutation:
    def apply(self, features: Dict[str, Any], config: Any, 
              feature_bank: Set[str], importance: Dict[str, float]) -> Dict[str, Any]:
        # Create a copy to avoid modifying the original dict
        result = dict(features)
        
        if random.random() < config.mutation_probs.get("metaboost", 0):
            for key in result:
                # Boost unimportant numeric features by 20%
                if isinstance(result[key], (int, float)) and (key not in importance or importance.get(key, 0) < 0.1):
                    result[key] = result[key] * 1.2
        return result