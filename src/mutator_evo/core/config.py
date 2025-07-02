# src/mutator_evo/core/config.py
from typing import Dict, Any

class DynamicConfig:
    def __init__(self, **kwargs):
        # Default parameters
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
        
        # State for adaptation
        self.performance_history = []
        self.mutation_success_rates = {op: [] for op in self._params["mutation_probs"]}
        
    def __getattr__(self, name: str) -> Any:
        if name in self._params:
            return self._params[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def update_based_on_performance(self, performance: float) -> None:
        """Update configuration based on performance"""
        self.performance_history.append(performance)
        
        # If performance hasn't improved in last 5 epochs, increase exploration
        if len(self.performance_history) >= 5:
            last_5 = self.performance_history[-5:]
            if all(x <= y for x, y in zip(last_5, last_5[1:])):
                for op in self.mutation_probs:
                    self._params["mutation_probs"][op] = min(0.5, self.mutation_probs[op] * 1.1)
    
    def record_mutation_success(self, operator: str, success: bool) -> None:
        """Record mutation success/failure for adaptation"""
        if operator not in self.mutation_success_rates:
            return
        self.mutation_success_rates[operator].append(success)
        
        # Keep only last 20 records
        if len(self.mutation_success_rates[operator]) > 20:
            self.mutation_success_rates[operator] = self.mutation_success_rates[operator][-20:]
        
        # Recalculate probability for this operator
        success_rate = sum(self.mutation_success_rates[operator]) / len(self.mutation_success_rates[operator])
        self._params["mutation_probs"][operator] = min(0.5, max(0.05, success_rate * 0.8))
    
    @property
    def params(self) -> Dict[str, Any]:
        """Return current configuration parameters"""
        return self._params.copy()