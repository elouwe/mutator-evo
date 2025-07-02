from __future__ import annotations

import json
import pathlib
import random
import uuid
import datetime
from collections import deque
from datetime import date
from typing import Any, Dict, List, Set, Optional

from .config import DynamicConfig
from .strategy_embedding import StrategyEmbedding
from ..operators.interfaces import IMutationOperator, IFeatureImportanceCalculator
from ..operators.mutation_impl import (
    DropMutation, AddMutation, ShiftMutation, 
    InvertMutation, MetabBoostMutation
)
from ..operators.importance_calculator import DefaultFeatureImportanceCalculator

class StrategyMutatorV2:
    def __init__(self, **kwargs):
        self.archival_pool: List[StrategyEmbedding] = []
        self.strategy_pool: List[StrategyEmbedding] = []
        self.frozen: bool = False
        self.equity_history: deque[float] = deque(maxlen=10)
        
        self.config = DynamicConfig(**kwargs)
        self._init_operators()
        
        self.checkpoint_every = 5
        self.checkpoint_dir = pathlib.Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self._tick = 0
        
        self.feature_bank: Set[str] = set()
        self.current_equity: float = 0.0
        self.step: int = 0
        self.rng = random.Random()
        self.rng.seed(42)
        self.importance: Dict[str, float] = {}

    def _init_operators(self):
        """Initialize mutation operators"""
        self.mutation_operators = [
            DropMutation(),
            AddMutation(),
            ShiftMutation(),
            InvertMutation(),
            MetabBoostMutation(),
        ]
        self.importance_calculator = DefaultFeatureImportanceCalculator()

    def _tweak(self, feats: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mutations to strategy features"""
        feats = dict(feats)  # Create a copy
        
        for operator in self.mutation_operators:
            try:
                feats = operator.apply(
                    features=feats,
                    config=self.config,
                    feature_bank=self.feature_bank,
                    importance=self.importance
                )
            except Exception as e:
                print(f"Operator error {type(operator).__name__}: {str(e)}")
                continue
                
        return feats

    def evolve(self) -> None:
        if self.frozen:
            print("[mutator] ⏸️ frozen – evolve() skipped")
            return

        if not self.strategy_pool:
            self._reinitialize_pool()
            return

        self.step += 1
        
        try:
            # 1) Sorting and selection
            self.strategy_pool.sort(key=lambda s: s.score(), reverse=True)
            top = self.strategy_pool[:self.config.top_k]
            
            # 2) Feature importance calculation
            self.importance = self.importance_calculator.compute(top)
            
            # 3) New strategy generation
            new_strategies = []
            for _ in range(self.config.n_mutants):
                # Parent selection
                if random.random() < self.config.mutation_probs["crossover"] and len(top) >= 2:
                    p1, p2 = random.sample(top, 2)
                    parent_feats = {**p1.features, **p2.features}
                else:
                    parent = random.choices(
                        top,
                        weights=[max(s.score(), 0.0) + 1e-3 for s in top]
                    )[0]
                    parent_feats = dict(parent.features)
                
                # Apply mutations
                mutant_feats = self._tweak(parent_feats)
                new_strategies.append(
                    StrategyEmbedding(
                        name=f"mut_{uuid.uuid4().hex[:8]}",
                        features=mutant_feats
                    )
                )
            
            # 4) Pool update
            self.strategy_pool = top + new_strategies
            
            # 5) Parameter adaptation
            if new_strategies:
                best_new_score = max(s.score() for s in new_strategies)
                self.config.update_based_on_performance(best_new_score)
            
            # 6) Logging
            best = top[0] if top else None
            print(
                f"[EVO] {date.today()} pool={len(self.strategy_pool)} "
                f"new={len(new_strategies)} best={best.name if best else '-'} "
                f"best_score={best.score() if best else 0:.2f}"
            )
            
            # 7) Checkpoint
            self._tick += 1
            if self._tick % self.checkpoint_every == 0:
                self.save_checkpoint()
                
        except Exception as e:
            print(f"Evolution cycle error: {str(e)}")
            self._reinitialize_pool()

    def _reinitialize_pool(self):
        """Reinitialize the strategy pool"""
        print("[mutator] Strategy pool is empty - creating new strategies")
        self.strategy_pool = [
            StrategyEmbedding.create_random(self.feature_bank)
            for _ in range(max(20, self.config.top_k * 2))
        ]

    def save_checkpoint(self, path: Optional[str] = None) -> None:
        """Save current state"""
        try:
            try:
                import dill
            except ImportError:
                import pickle as dill
                
            path = pathlib.Path(path or f"mutator_checkpoint_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.pkl")
            
            with path.open("wb") as f:
                dill.dump({
                    "pool": self.strategy_pool,
                    "archive": self.archival_pool,
                    "config_params": self.config.params,
                    "feature_bank": list(self.feature_bank),
                    "current_equity": self.current_equity,
                }, f)
            print(f"💾 checkpoint → {path}")
            
        except Exception as e:
            print(f"Save error: {str(e)}")

    @classmethod
    def load_checkpoint(cls, path: str) -> "StrategyMutatorV2":
        """Load saved state"""
        try:
            try:
                import dill
            except ImportError:
                import pickle as dill
                
            with pathlib.Path(path).open("rb") as f:
                state = dill.load(f)
                
            self = cls(**state.get("config_params", {}))
            self.archival_pool = state.get("archive", [])
            self.strategy_pool = state.get("pool", [])
            self.feature_bank = set(state.get("feature_bank", set()))
            self.current_equity = state.get("current_equity", 0.0)
            
            if "config_params" in state:
                self.config = DynamicConfig(**state["config_params"])
                
            return self
            
        except Exception as e:
            print(f"Load error: {str(e)}")
            return cls()  # Return new mutator if error occurs

    def freeze(self) -> None:
        self.frozen = True

    def unfreeze(self) -> None:
        self.frozen = False