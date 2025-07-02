# src/mutator_evo/core/strategy_mutator.py
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
        
        self.checkpoint_every = kwargs.get('checkpoint_every', 5)
        self.checkpoint_dir = pathlib.Path(kwargs.get('checkpoint_dir', "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
        for operator in self.mutation_operators:
            try:
                feats = operator.apply(
                    feats, 
                    self.config, 
                    self.feature_bank, 
                    self.importance
                )
            except Exception as e:
                print(f"Error in operator {type(operator).__name__}: {str(e)}")
                continue
        return feats

    def evolve(self) -> None:
        if self.frozen:
            print("[mutator] ⏸️ frozen - evolve() skipped")
            return

        if not self.strategy_pool:
            self._reinitialize_pool()
            return

        self.step += 1
        
        try:
            # 1) Sort pool by scores
            self.strategy_pool.sort(key=lambda s: s.score(), reverse=True)
            top_k = min(self.config.top_k, len(self.strategy_pool))
            top = self.strategy_pool[:top_k]

            # 2) Calculate feature importance
            self.importance = self.importance_calculator.compute(top)

            # 3) Trim pool (elitism)
            self.strategy_pool = top.copy()

            # 4) Generate mutants
            new = []
            for _ in range(self.config.n_mutants):
                if (
                    random.random() < self.config.mutation_probs["crossover"]
                    and len(top) >= 2
                ):
                    p1, p2 = random.sample(top, 2)
                    parent_feats = {**p1.features, **p2.features}
                else:
                    weights = [max(s.score(), 0.0) + 1e-3 for s in top]
                    parent = random.choices(top, weights=weights)[0]
                    parent_feats = dict(parent.features)

                mutant_feats = self._tweak(parent_feats)
                name = f"mut_{uuid.uuid4().hex[:8]}"

                new_strat = StrategyEmbedding(
                    name=name,
                    features=mutant_feats
                )
                new.append(new_strat)

            self.strategy_pool.extend(new)

            # 5) Update config based on performance
            if new:
                best_new_score = max(s.score() for s in new)
                self.config.update_based_on_performance(best_new_score)

            # 6) Logging
            best = self.strategy_pool[0] if self.strategy_pool else None
            print(
                f"[EVO] {date.today()}  pool={len(self.strategy_pool)}  "
                f"new={len(new)}  best={best.name if best else '—'} "
                f"best_score={best.score() if best else 0:.2f}"
            )

            self._tick += 1
            if self._tick % self.checkpoint_every == 0:
                self.save_checkpoint()
                
        except Exception as e:
            print(f"Error in evolution cycle: {str(e)}")
            self._reinitialize_pool()

    def _reinitialize_pool(self):
        """Reinitialize strategy pool when empty"""
        print("[mutator] Strategy pool empty - creating new strategies")
        self.strategy_pool = [
            StrategyEmbedding.create_random(self.feature_bank)
            for _ in range(max(20, self.config.top_k * 2))
        ]

    def save_checkpoint(self, path: Optional[str] = None) -> None:
        try:
            try:
                import dill
            except ImportError:
                import pickle as dill
                
            # Generate timestamped filename
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"mutator_checkpoint_{ts}.pkl"
            path = self.checkpoint_dir / filename
            
            with path.open("wb") as f:
                dill.dump(
                    {
                        "pool": self.strategy_pool,
                        "archive": self.archival_pool,
                        "config_params": self.config.params,
                        "feature_bank": list(self.feature_bank),
                        "current_equity": self.current_equity,
                        "importance": self.importance,  # Include feature importance
                    },
                    f,
                )
            print(f"💾 Checkpoint saved → {path}")
            
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")

    @classmethod
    def load_checkpoint(cls, path: str) -> "StrategyMutatorV2":
        try:
            try:
                import dill
            except ImportError:
                import pickle as dill
                
            with pathlib.Path(path).open("rb") as f:
                state = dill.load(f)

            # Create instance with saved config params
            config_params = state.get("config_params", {})
            self = cls(**config_params)
            
            # Restore state
            self.archival_pool = state.get("archive", [])
            self.strategy_pool = state.get("pool", [])
            self.feature_bank = set(state.get("feature_bank", set()))
            self.current_equity = state.get("current_equity", 0.0)
            self.importance = state.get("importance", {})
            
            return self
            
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            # Return new mutator on error
            return cls()

    def freeze(self) -> None:
        self.frozen = True

    def unfreeze(self) -> None:
        self.frozen = False