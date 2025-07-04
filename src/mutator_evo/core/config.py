# src/mutator_evo/core/config.py
import math
from typing import Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)

class DynamicConfig:
    def __init__(self, **kwargs):
        # Default parameters with increased mutation probabilities
        self._params = {
            "top_k": 10,
            "n_mutants": 10,
            "max_age": 10,
            "drop_threshold": 0.7,
            "mutation_probs": {
                "add": 0.50,
                "drop": 0.40,
                "shift": 0.35,
                "invert": 0.25,
                "metaboost": 0.20,
                "crossover": 0.50,
                "uniform_crossover": 0.45,
                "rl_mutation": 0.35,
            },
            **kwargs
        }
        
        # State for adaptation
        self.performance_history = []
        self.top_strategies = []
        
        # Initialize operator stats for UCB
        self.operator_stats = {
            "add": {"n": 0, "total_impact": 0, "min_impact": 0},
            "drop": {"n": 0, "total_impact": 0, "min_impact": 0},
            "shift": {"n": 0, "total_impact": 0, "min_impact": 0},
            "invert": {"n": 0, "total_impact": 0, "min_impact": 0},
            "metaboost": {"n": 0, "total_impact": 0, "min_impact": 0},
            "uniform_crossover": {"n": 0, "total_impact": 0, "min_impact": 0},
            "rl_mutation": {"n": 0, "total_impact": 0, "min_impact": 0}
        }
        self.ucb_c = 1.0
        
        # Mapping from class names to short keys
        self.operator_name_map = {
            "AddMutation": "add",
            "DropMutation": "drop",
            "ShiftMutation": "shift",
            "InvertMutation": "invert",
            "MetabBoostMutation": "metaboost",
            "UniformCrossover": "uniform_crossover",
            "RLMutation": "rl_mutation"
        }
        
    def __getattr__(self, name: str) -> Any:
        if name in self._params:
            return self._params[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def update_based_on_performance(self, performance: float) -> None:
        self.performance_history.append(performance)
        
        # Reset if performance plateaus
        if len(self.performance_history) > 10:
            last_10 = self.performance_history[-10:]
            if max(last_10) - min(last_10) < 0.5:
                for op in self.mutation_probs:
                    self._params["mutation_probs"][op] = min(0.9, self.mutation_probs[op] * 1.2)
    
    @property
    def params(self) -> Dict[str, Any]:
        return self._params.copy()

    def select_operator(self):
        total_trials = sum(stats['n'] for stats in self.operator_stats.values())
        
        # If any operator hasn't been tried, try it first
        for op, stats in self.operator_stats.items():
            if stats['n'] == 0:
                return op

        # Calculate UCB for each operator
        ucb_values = {}
        for op, stats in self.operator_stats.items():
            if stats['n'] > 0:
                avg_impact = stats['total_impact'] / stats['n']
            else:
                avg_impact = 0
                
            exploration_term = self.ucb_c * math.sqrt(math.log(total_trials) / stats['n']) if stats['n'] > 0 else 1.0
            ucb_values[op] = avg_impact + exploration_term

        return max(ucb_values, key=ucb_values.get)

    def update_operator_stats(self, operator: str, impact: float):
        op_short = self.operator_name_map.get(operator)
        if op_short and op_short in self.operator_stats:
            self.operator_stats[op_short]["n"] += 1
            self.operator_stats[op_short]["total_impact"] += impact
            self.operator_stats[op_short]["min_impact"] = min(
                self.operator_stats[op_short].get("min_impact", 0), 
                impact
            )
            return
        
        # Try to match by class name prefix
        for class_name, short in self.operator_name_map.items():
            if class_name in operator:
                if short in self.operator_stats:
                    self.operator_stats[short]["n"] += 1
                    self.operator_stats[short]["total_impact"] += impact
                    self.operator_stats[short]["min_impact"] = min(
                        self.operator_stats[short].get("min_impact", 0), 
                        impact
                    )
                    return
        
        # If not found, create new entry
        logger.warning(f"Adding new operator to stats: {operator}")
        self.operator_stats[operator] = {"n": 1, "total_impact": impact, "min_impact": impact}

    def adapt_mutation_probs(self):
        """Adapt based on average impact score"""
        BASE_PROB = 0.3
        MAX_PROB = 0.9
        
        logger.debug("\nAdapting mutation probabilities:")
        
        # Collect all average impacts
        avg_impacts = []
        for stats in self.operator_stats.values():
            if stats['n'] > 0:
                avg_impact = stats['total_impact'] / stats['n']
                avg_impacts.append(avg_impact)
        
        # Calculate min/max impact only if we have data
        if avg_impacts:
            min_impact = min(avg_impacts)
            max_impact = max(avg_impacts)
        else:
            min_impact = 0
            max_impact = 1
        
        for op_short, stats in self.operator_stats.items():
            if stats['n'] < 3:
                continue
                
            # Calculate normalized impact
            avg_impact = stats['total_impact'] / stats['n']
            normalized_impact = (avg_impact - min_impact) / (max_impact - min_impact + 1e-8)
                
            # New probability = base + scaled impact
            new_prob = BASE_PROB + 0.6 * normalized_impact
            new_prob = min(MAX_PROB, new_prob)
            
            # Apply only to operators in mutation_probs
            if op_short in self._params["mutation_probs"]:
                current_prob = self._params["mutation_probs"][op_short]
                # Smooth update
                smoothed_prob = 0.7 * current_prob + 0.3 * new_prob
                self._params["mutation_probs"][op_short] = smoothed_prob
                logger.debug(f"  {op_short}: impact={avg_impact:.3f} prob: {current_prob:.2f} -> {smoothed_prob:.2f}")