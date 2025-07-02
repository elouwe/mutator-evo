from abc import ABC, abstractmethod
from typing import Any, Dict, Set

class IMutationOperator(ABC):
    @abstractmethod
    def apply(self, features: Dict[str, Any], config: Any, 
              feature_bank: Set[str], importance: Dict[str, float]) -> Dict[str, Any]:
        pass

class IFeatureImportanceCalculator(ABC):
    @abstractmethod
    def compute(self, top_strats: list) -> Dict[str, float]:
        pass