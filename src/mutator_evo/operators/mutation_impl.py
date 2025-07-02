import random
from typing import Any, Dict, Set

class DropMutation:
    def apply(self, features: Dict[str, Any], config: Any, 
              feature_bank: Set[str], importance: Dict[str, float]) -> Dict[str, Any]:
        if random.random() < config.mutation_probs.get("drop", 0) and len(features) > 1:
            # Remove the least important features
            if importance:
                to_drop = min(features.keys(), key=lambda k: importance.get(k, 0))
            else:
                to_drop = random.choice(list(features.keys()))
            features.pop(to_drop)
        return features

class AddMutation:
    def apply(self, features: Dict[str, Any], config: Any, 
              feature_bank: Set[str], importance: Dict[str, float]) -> Dict[str, Any]:
        if random.random() < config.mutation_probs.get("add", 0):
            available = feature_bank - set(features.keys())
            if available:
                # Adding the most important features
                if importance:
                    to_add = max(available, key=lambda k: importance.get(k, 0))
                else:
                    to_add = random.choice(list(available))
                features[to_add] = random.random()
        return features

class ShiftMutation:
    def apply(self, features: Dict[str, Any], config: Any, 
              feature_bank: Set[str], importance: Dict[str, float]) -> Dict[str, Any]:
        for k in list(features.keys()):
            if isinstance(features[k], (int, float)) and random.random() < config.mutation_probs.get("shift", 0):
                features[k] = features[k] * random.uniform(0.8, 1.2)
        return features

class InvertMutation:
    def apply(self, features: Dict[str, Any], config: Any, 
              feature_bank: Set[str], importance: Dict[str, float]) -> Dict[str, Any]:
        for k in list(features.keys()):
            if isinstance(features[k], bool) and random.random() < config.mutation_probs.get("invert", 0):
                features[k] = not features[k]
        return features

class MetabBoostMutation:
    def apply(self, features: Dict[str, Any], config: Any, 
              feature_bank: Set[str], importance: Dict[str, float]) -> Dict[str, Any]:
        if random.random() < config.mutation_probs.get("metaboost", 0):
            for k in features:
                if isinstance(features[k], (int, float)) and (k not in importance or importance[k] < 0.1):
                    features[k] = features[k] * 1.2
        return features