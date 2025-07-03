# src/mutator_evo/operators/mutation_impl.py
import random
from typing import Dict, Any, Set

class DropMutation:
    def apply(self, features: Dict[str, Any], config: Any, 
              feature_bank: Set[str], importance: Dict[str, float]) -> Dict[str, Any]:
        result = dict(features)
        
        # Only drop if we have at least 3 features
        if len(result) > 3 and random.random() < config.mutation_probs.get("drop", 0):
            # Never drop critical parameters
            protected = {"trade_size", "stop_loss", "take_profit"}
            candidates = [k for k in result.keys() 
                          if k not in protected and k != "rl_agent"]
            
            if candidates:
                # Remove least important features first
                if importance:
                    candidate_importance = {k: importance.get(k, 0) for k in candidates}
                    min_imp = min(candidate_importance.values())
                    to_drop = random.choice([k for k, v in candidate_importance.items() 
                                            if v == min_imp])
                else:
                    to_drop = random.choice(candidates)
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
                    to_add = random.choice(sorted_available) if sorted_available else random.choice(list(available))
                else:
                    to_add = random.choice(list(available))
                
                # Set meaningful initial value
                if 'period' in to_add:
                    result[to_add] = random.randint(5, 25)  # Shorter periods
                elif 'trade_size' in to_add:
                    result[to_add] = random.uniform(0.15, 0.4)  # Larger position size
                elif 'stop_loss' in to_add:
                    result[to_add] = random.uniform(0.01, 0.04)  # 1-4% stop loss
                elif 'take_profit' in to_add:
                    result[to_add] = random.uniform(0.03, 0.08)  # 3-8% take profit
                else:
                    result[to_add] = True  # Default for boolean features
        return result

class ShiftMutation:
    def apply(self, features: Dict[str, Any], config: Any, 
              feature_bank: Set[str], importance: Dict[str, float]) -> Dict[str, Any]:
        result = dict(features)
        
        # Only shift 1 parameter per mutation
        if random.random() < config.mutation_probs.get("shift", 0):
            # Only modify numeric features (skip booleans and RL agent)
            numeric_keys = [k for k, v in result.items() 
                           if isinstance(v, (int, float)) and not isinstance(v, bool)
                           and not k.startswith("rl_agent")]
            
            if numeric_keys:
                key = random.choice(numeric_keys)
                original = result[key]
                
                # Apply moderate random shift (±15% original value)
                shift_factor = 0.85 + random.random() * 0.3  # 0.85x to 1.15x
                result[key] = original * shift_factor
                
                # Ensure minimum values for periods
                if 'period' in key:
                    result[key] = max(5, min(50, result[key]))
        return result

class InvertMutation:
    def apply(self, features: Dict[str, Any], config: Any, 
              feature_bank: Set[str], importance: Dict[str, float]) -> Dict[str, Any]:
        result = dict(features)
        
        # Only invert 1 parameter per mutation
        if random.random() < config.mutation_probs.get("invert", 0):
            boolean_keys = [k for k, v in result.items() 
                           if isinstance(v, bool) and not k.startswith("rl_agent")]
            
            if boolean_keys:
                key = random.choice(boolean_keys)
                result[key] = not result[key]
        return result

class MetabBoostMutation:
    def apply(self, features: Dict[str, Any], config: Any, 
              feature_bank: Set[str], importance: Dict[str, float]) -> Dict[str, Any]:
        result = dict(features)
        
        if random.random() < config.mutation_probs.get("metaboost", 0):
            # Only boost 1 parameter per mutation
            numeric_keys = [k for k, v in result.items() 
                           if isinstance(v, (int, float)) and not isinstance(v, bool)
                           and not k.startswith("rl_agent")]
            
            if numeric_keys:
                key = random.choice(numeric_keys)
                original = result[key]
                
                # Apply moderate boost (0.8x to 1.5x)
                boost_factor = 0.8 + random.random() * 0.7
                result[key] = original * boost_factor
                
                # Ensure reasonable bounds
                if 'period' in key:
                    result[key] = max(5, min(50, result[key]))
                elif 'trade_size' in key:
                    result[key] = max(0.1, min(0.5, result[key]))
                elif 'stop_loss' in key:
                    result[key] = max(0.005, min(0.1, result[key]))
                elif 'take_profit' in key:
                    result[key] = max(0.01, min(0.15, result[key]))
        return result