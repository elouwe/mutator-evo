from types import SimpleNamespace
from typing import Dict, Any

class DynamicConfig:
    def __init__(self, **kwargs):
        self._params = {
            "top_k": 5,
            "n_mutants": 3,
            "max_age": 15,
            "drop_threshold": 0.8,
            "mutation_probs": {
                "add": 0.30,
                "drop": 0.25,
                "shift": 0.15,
                "invert": 0.10,
                "metaboost": 0.05,
                "crossover": 0.30,
            },
            **kwargs
        }
        
        self.performance_history = []
        self.mutation_success_rates = {op: [] for op in self._params["mutation_probs"]}
        
    def __getattr__(self, name):
        if name in self._params:
            return self._params[name]
        raise AttributeError(f"No such attribute: {name}")
    
    def update_based_on_performance(self, performance: float):
        """Performance-based parameter adaptation"""
        self.performance_history.append(performance)
        
        if len(self.performance_history) > 5:
            last_5 = self.performance_history[-5:]
            if max(last_5) - min(last_5) < 0.1:  # Plateau
                for op in self._params["mutation_probs"]:
                    self._params["mutation_probs"][op] = min(0.5, self._params["mutation_probs"][op] * 1.1)
    
    def record_mutation_success(self, operator: str, success: bool):
        """Recording the success of mutations"""
        if operator in self.mutation_success_rates:
            self.mutation_success_rates[operator].append(success)
            if len(self.mutation_success_rates[operator]) > 20:
                self.mutation_success_rates[operator] = self.mutation_success_rates[operator][-20:]
            
            success_rate = sum(self.mutation_success_rates[operator]) / len(self.mutation_success_rates[operator])
            self._params["mutation_probs"][operator] = max(0.05, min(0.5, success_rate * 0.8))
    
    @property
    def params(self) -> Dict[str, Any]:
        return self._params.copy()