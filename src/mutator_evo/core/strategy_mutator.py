# src/mutator_evo/core/strategy_mutator.py
from __future__ import annotations

import json
import pathlib
import random
import uuid
import datetime
import logging
import numpy as np
from collections import deque, defaultdict
from datetime import date
from typing import Any, Dict, List, Set, Optional, Tuple

from .config import DynamicConfig
from .strategy_embedding import StrategyEmbedding
from ..operators.interfaces import IMutationOperator, IFeatureImportanceCalculator
from ..operators.mutation_impl import (
    DropMutation, AddMutation, ShiftMutation, 
    InvertMutation, MetabBoostMutation
)
from ..operators.importance_calculator import (
    DefaultFeatureImportanceCalculator,
    HybridFeatureImportanceCalculator
)

# Импорт новых операторов с обработкой ошибок
try:
    from ..operators._uniform_crossover import UniformCrossover
    from ..operators.rl_mutation import RLMutation
except ImportError as e:
    print(f"Warning: Could not import new operators: {e}")
    # Создаем заглушки для операторов
    class UniformCrossover:
        def apply(self, features, config, feature_bank, importance):
            return features
    
    class RLMutation:
        def apply(self, features, config, feature_bank, importance):
            return features

logger = logging.getLogger(__name__)

class StrategyMutatorV2:
    def __init__(self, backtest_adapter=None, use_shap=True, **kwargs):
        self.archival_pool: List[StrategyEmbedding] = []
        self.strategy_pool: List[StrategyEmbedding] = []
        self.frozen: bool = False
        self.equity_history: deque[float] = deque(maxlen=10)
        
        self.config = DynamicConfig(**kwargs)
        self.use_shap = use_shap
        self._init_operators()
        
        self.checkpoint_every = kwargs.get('checkpoint_every', 5)
        self.max_checkpoints = kwargs.get('max_checkpoints', 5)
        self.checkpoint_dir = pathlib.Path(kwargs.get('checkpoint_dir', "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._tick = 0
        
        self.feature_bank: Set[str] = set()
        self.current_equity: float = 0.0
        self.step: int = 0
        self.rng = random.Random()
        self.rng.seed(42)
        self.importance: Dict[str, float] = {}
        self.backtest_adapter = backtest_adapter
        
        # Track operator usage for current generation
        self.operator_usage = []
        # Track mutants for next generation evaluation: {mutant_name: (operator_name, parent_name)}
        self.pending_evaluation = {}

    def _init_operators(self):
        self.mutation_operators = [
            DropMutation(),
            AddMutation(),
            ShiftMutation(),
            InvertMutation(),
            MetabBoostMutation(),
            UniformCrossover(),
            RLMutation(),
        ]
        
        # Create mapping from class names to short names
        self.operator_short_names = {
            "AddMutation": "add",
            "DropMutation": "drop",
            "ShiftMutation": "shift",
            "InvertMutation": "invert",
            "MetabBoostMutation": "metaboost",
            "UniformCrossover": "uniform_crossover",
            "RLMutation": "rl_mutation"
        }
        
        # Выбираем калькулятор важности признаков
        if self.use_shap:
            self.importance_calculator = HybridFeatureImportanceCalculator()
        else:
            self.importance_calculator = DefaultFeatureImportanceCalculator()

    def _tweak(self, feats: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Apply one mutation operator selected by UCB and return the operator name."""
        if not self.mutation_operators:
            return feats, "no_operator"

        # Select operator with UCB
        op_short = self.config.select_operator()
        
        # Apply the selected operator
        selected_operator = None
        for op in self.mutation_operators:
            class_name = op.__class__.__name__
            short_name = self.operator_short_names.get(class_name, "")
            if short_name == op_short:
                selected_operator = op
                break
        
        # Fallback if not found
        if selected_operator is None:
            logger.warning(f"Operator {op_short} not found, using random")
            selected_operator = random.choice(self.mutation_operators)

        try:
            new_feats = selected_operator.apply(
                feats, self.config, self.feature_bank, self.importance
            )
            # Return class name for stats tracking
            return new_feats, selected_operator.__class__.__name__
        except Exception as e:
            logger.error(f"Operator error: {type(selected_operator).__name__}: {str(e)}")
            return feats, selected_operator.__class__.__name__

    def evolve(self) -> None:
        if self.frozen:
            logger.info("[mutator] Skipped: frozen")
            return

        if not self.strategy_pool:
            self._reinitialize_pool()
            return

        self.step += 1
        
        try:
            # Step 1: Evaluate pending mutants from previous generation
            if self.pending_evaluation:
                logger.info("Evaluating pending operators from previous generation...")
                
                # Calculate baseline scores
                parent_scores = {}
                for mutant_name, (op_name, parent_name) in list(self.pending_evaluation.items()):
                    parent = next((s for s in self.strategy_pool if s.name == parent_name), None)
                    if parent:
                        parent_scores[parent_name] = parent.score()
                
                for mutant_name, (op_name, parent_name) in list(self.pending_evaluation.items()):
                    mutant = next((s for s in self.strategy_pool if s.name == mutant_name), None)
                    parent_score = parent_scores.get(parent_name, 0)
                    
                    if mutant is None:
                        # Default impact if mutant is missing
                        impact = 0
                        logger.info(f"Operator {op_name} for {mutant_name}: mutant missing, impact=0.0")
                    else:
                        mutant_score = mutant.score()
                        
                        # Calculate impact as improvement over parent
                        improvement = mutant_score - parent_score
                        
                        # Calculate relative improvement factor
                        if parent_score != 0:
                            relative_improvement = improvement / abs(parent_score)
                        else:
                            relative_improvement = improvement
                            
                        # Combined impact metric
                        impact = np.clip(improvement + relative_improvement, -2, 2)
                        logger.info(f"Operator {op_name} for {mutant_name}: "
                                    f"parent_score={parent_score:.2f} mutant_score={mutant_score:.2f} "
                                    f"impact={impact:.3f}")
                    
                    # Update operator stats with impact score
                    self.config.update_operator_stats(op_name, impact)
                    del self.pending_evaluation[mutant_name]
            
            # Step 2: Backtest new strategies
            new_strategies = [s for s in self.strategy_pool if s.oos_metrics is None]
            if new_strategies and self.backtest_adapter:
                logger.info(f"Backtesting {len(new_strategies)} strategies...")
                for strategy in new_strategies:
                    try:
                        results = self.backtest_adapter.evaluate(strategy)
                        strategy.oos_metrics = results
                        logger.info(f"  {strategy.name}: score={strategy.score():.2f}, trades={results.get('trade_count', 0)}")
                    except Exception as e:
                        logger.error(f"Backtest failed for {strategy.name}: {str(e)}")
                        strategy.oos_metrics = {
                            "oos_sharpe": -5.0,
                            "oos_max_drawdown": 100.0,
                            "oos_win_rate": 0.0,
                            "overfitting_penalty": 1.0,
                            "trade_count": 0
                        }
                        strategy._cached_score = -5.0
            
            # Step 3: Sort and select top strategies
            self.strategy_pool.sort(key=lambda s: s.score(), reverse=True)
            
            if not self.strategy_pool:
                logger.warning("Strategy pool empty, reinitializing")
                self._reinitialize_pool()
                return
                
            top_k = min(self.config.top_k, len(self.strategy_pool))
            top = self.strategy_pool[:top_k]

            # Calculate feature importance
            try:
                method = "SHAP" if self.use_shap else "Default"
                logger.info(f"Calculating feature importance using {method} method...")
                
                self.importance = self.importance_calculator.compute(top)
                
                # Log top features
                if self.importance:
                    sorted_importance = sorted(
                        self.importance.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:5]
                    logger.info("Top features:")
                    for feat, imp in sorted_importance:
                        logger.info(f"  {feat}: {imp:.4f}")
            except Exception as e:
                logger.error(f"Feature importance error: {str(e)}")
                # Fallback to default method
                self.importance = DefaultFeatureImportanceCalculator().compute(top)

            # Elitism selection - keep top strategies
            self.strategy_pool = top.copy()

            # Pass top strategies to config for crossover operator
            self.config.top_strategies = top

            # Step 4: Generate mutants
            new = []
            self.operator_usage = []  # Reset for new generation
            
            for _ in range(self.config.n_mutants):
                # Higher chance of crossover
                if random.random() < self.config.mutation_probs["crossover"] and len(top) >= 2:
                    p1, p2 = random.sample(top, 2)
                    parent_feats = {**p1.features, **p2.features}
                    parent_name = f"{p1.name}+{p2.name}"
                else:
                    weights = [max(s.score(), 0.01) for s in top]  # Avoid zero weights
                    parent = random.choices(top, weights=weights)[0]
                    parent_feats = dict(parent.features)
                    parent_name = parent.name

                mutant_feats, op_name = self._tweak(parent_feats)
                name = f"mut_{uuid.uuid4().hex[:8]}"
                new_strat = StrategyEmbedding(name=name, features=mutant_feats)
                new.append(new_strat)
                
                # Track for next evaluation (store both operator and parent)
                self.pending_evaluation[name] = (op_name, parent_name)
                self.operator_usage.append((op_name, name))

            # Add new mutants to pool
            self.strategy_pool.extend(new)

            # Step 5: Adapt mutation probabilities
            self.config.adapt_mutation_probs()

            # Log operator stats
            logger.info("\nOperator Statistics:")
            for op, stats in self.config.operator_stats.items():
                if stats['n'] > 0:
                    avg_impact = stats['total_impact'] / stats['n']
                    prob = self.config.mutation_probs.get(op, 0.0)
                    logger.info(f"  {op}: n={stats['n']} avg_impact={avg_impact:.3f} prob={prob:.2f}")

            # Update config based on performance
            if new:
                best_new_score = max(s.score() for s in new) if new else 0
                self.config.update_based_on_performance(best_new_score)

            # Logging
            best = self.strategy_pool[0] if self.strategy_pool else None
            logger.info(f"[EVO] Gen {self.step}: pool={len(self.strategy_pool)} "
                  f"best={best.name if best else 'N/A'} score={best.score() if best else 0:.2f}")

            # Checkpoint
            self._tick += 1
            if self._tick % self.checkpoint_every == 0:
                self.save_checkpoint()
                
        except Exception as e:
            logger.error(f"Evolution error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self._reinitialize_pool()

    def _reinitialize_pool(self):
        logger.info("[mutator] Reinitializing strategy pool")
        self.strategy_pool = [
            StrategyEmbedding.create_random(self.feature_bank)
            for _ in range(30)  # Larger initial pool
        ]
        self.pending_evaluation = {}

    def save_checkpoint(self, path: Optional[str] = None) -> None:
        try:
            import dill
        except ImportError:
            import pickle as dill
            
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"mutator_checkpoint_{ts}.pkl"
        path = self.checkpoint_dir / filename if path is None else pathlib.Path(path)
        
        try:
            with path.open("wb") as f:
                dill.dump({
                    "pool": self.strategy_pool,
                    "archive": self.archival_pool,
                    "config_params": self.config.params,
                    "feature_bank": list(self.feature_bank),
                    "current_equity": self.current_equity,
                    "importance": self.importance,
                    "operator_stats": self.config.operator_stats,
                    "pending_evaluation": self.pending_evaluation,
                }, f)
            logger.info(f"Checkpoint saved: {path}")
            
            # Auto-clean old checkpoints
            if self.max_checkpoints > 0:
                checkpoint_files = sorted(
                    self.checkpoint_dir.glob("mutator_checkpoint_*.pkl"),
                    key=lambda f: f.stat().st_mtime,
                    reverse=True
                )
                # Remove all but the last N checkpoints
                for old_checkpoint in checkpoint_files[self.max_checkpoints:]:
                    try:
                        old_checkpoint.unlink()
                        logger.info(f"Removed old checkpoint: {old_checkpoint}")
                    except Exception as e:
                        logger.error(f"Failed to remove {old_checkpoint}: {str(e)}")
        except Exception as e:
            logger.error(f"Save error: {str(e)}")

    @classmethod
    def load_checkpoint(cls, path: str) -> "StrategyMutatorV2":
        try:
            import dill
        except ImportError:
            import pickle as dill
            
        try:
            with pathlib.Path(path).open("rb") as f:
                state = dill.load(f)

            config_params = state.get("config_params", {})
            self = cls(**config_params)
            
            self.archival_pool = state.get("archive", [])
            self.strategy_pool = state.get("pool", [])
            self.feature_bank = set(state.get("feature_bank", set()))
            self.current_equity = state.get("current_equity", 0.0)
            self.importance = state.get("importance", {})
            self.pending_evaluation = state.get("pending_evaluation", {})
            
            # Load operator stats
            operator_stats = state.get("operator_stats", {})
            for op, stats in operator_stats.items():
                if op in self.config.operator_stats:
                    self.config.operator_stats[op] = stats
            
            return self
        except Exception as e:
            logger.error(f"Load error: {str(e)}")
            return cls()

    def freeze(self) -> None:
        self.frozen = True

    def unfreeze(self) -> None:
        self.frozen = False