# src/mutator_evo/operators/importance_calculator.py
from collections import Counter, defaultdict
from typing import List, Dict

class DefaultFeatureImportanceCalculator:
    def compute(self, top_strats: List) -> Dict[str, float]:
        freq = Counter(f for s in top_strats for f in s.features)
        local_gain = defaultdict(float)

        for s in top_strats:
            base = s.score()
            for f in s.features:
                tmp_score = s.estimate_score_without(f)
                local_gain[f] += max(base - tmp_score, 0)

        max_lg = max(local_gain.values(), default=1)
        max_fr = max(freq.values(), default=1)

        return {
            f: 0.5 * (local_gain[f]/max_lg) + 0.5 * (freq[f]/max_fr)
            for f in freq
        }