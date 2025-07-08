# src/mutator_evo/core/strategy_mutator.py
from __future__ import annotations

import json
import pathlib
import random
import uuid
import datetime
import logging
import numpy as np
import ray
import time
from collections import deque, defaultdict
from datetime import date
from typing import Any, Dict, List, Set, Optional, Tuple
import pandas as pd
import backtrader as bt
import zlib
import hashlib

from .config import DynamicConfig
from .ray_pool import RayPool
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

# Import new operators with error handling
try:
    from ..operators._uniform_crossover import UniformCrossover
    from ..operators.rl_mutation import RLMutation
except ImportError as e:
    print(f"Warning: Could not import new operators: {e}")
    # Create stubs for operators
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
        self.use_ray = True
        self.backtest_time = 0.0
        
        self.operator_usage = []
        self.pending_evaluation = {}
        self.rl_usage = 0.0

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
        
        self.operator_short_names = {
            "AddMutation": "add",
            "DropMutation": "drop",
            "ShiftMutation": "shift",
            "InvertMutation": "invert",
            "MetabBoostMutation": "metaboost",
            "UniformCrossover": "uniform_crossover",
            "RLMutation": "rl_mutation"
        }
        
        if self.use_shap:
            self.importance_calculator = HybridFeatureImportanceCalculator()
        else:
            self.importance_calculator = DefaultFeatureImportanceCalculator()

    def _tweak(self, feats: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        if not self.mutation_operators:
            return feats, "no_operator"

        op_short = self.config.select_operator()
        
        selected_operator = None
        for op in self.mutation_operators:
            class_name = op.__class__.__name__
            short_name = self.operator_short_names.get(class_name, "")
            if short_name == op_short:
                selected_operator = op
                break
        
        if selected_operator is None:
            logger.warning(f"Operator {op_short} not found, using random")
            selected_operator = random.choice(self.mutation_operators)

        try:
            new_feats = selected_operator.apply(
                feats, self.config, self.feature_bank, self.importance
            )
            return new_feats, selected_operator.__class__.__name__
        except Exception as e:
            logger.error(f"Operator error: {type(selected_operator).__name__}: {str(e)}")
            return feats, selected_operator.__class__.__name__

    def evolve(self) -> None:
        if self.frozen:
            logger.info("[mutator] Skipped: frozen")
            return

        # Check for degeneration: if all strategies have score < -2.0
        if self.strategy_pool and all(s.score() < -2.0 for s in self.strategy_pool):
            logger.warning("Pool degradation detected! Reinitializing with enhanced strategies...")
            self._reinitialize_pool()
            return

        if not self.strategy_pool:
            self._reinitialize_pool()
            return

        self.step += 1
        start_time = time.time()
        
        try:
            # Step 1: Decompress all strategies
            for s in self.strategy_pool:
                s.decompress()

            # Step 2: Evaluate pending mutants from previous generation
            if self.pending_evaluation:
                logger.info("Evaluating pending operators from previous generation...")
                
                parent_scores = {}
                for mutant_name, (op_name, parent_name) in list(self.pending_evaluation.items()):
                    parent = next((s for s in self.strategy_pool if s.name == parent_name), None)
                    if parent:
                        parent_scores[parent_name] = parent.score()
                
                for mutant_name, (op_name, parent_name) in list(self.pending_evaluation.items()):
                    mutant = next((s for s in self.strategy_pool if s.name == mutant_name), None)
                    parent_score = parent_scores.get(parent_name, 0)
                    
                    if mutant is None:
                        impact = 0
                        logger.info(f"Operator {op_name} for {mutant_name}: mutant missing, impact=0.0")
                    else:
                        mutant_score = mutant.score()
                        improvement = mutant_score - parent_score
                        if parent_score != 0:
                            relative_improvement = improvement / abs(parent_score)
                        else:
                            relative_improvement = improvement
                        impact = np.clip(improvement + relative_improvement, -2, 2)
                        logger.info(f"Operator {op_name} for {mutant_name}: "
                                    f"parent_score={parent_score:.2f} mutant_score={mutant_score:.2f} "
                                    f"impact={impact:.3f}")
                    
                    self.config.update_operator_stats(op_name, impact)
                    del self.pending_evaluation[mutant_name]
            
            # Step 3: Backtest new strategies
            new_strategies = [s for s in self.strategy_pool if s.oos_metrics is None]
            if new_strategies and self.backtest_adapter:
                logger.info(f"Backtesting {len(new_strategies)} strategies...")
                backtest_start = time.time()
                
                market_data = self.backtest_adapter.original_df
                
                if self.use_ray:
                    logger.info("Using Ray for parallel backtesting")
                    batch_size = max(4, len(new_strategies) // 4)
                    batches = [new_strategies[i:i + batch_size] 
                              for i in range(0, len(new_strategies), batch_size)]
                    
                    with RayPool().pool() as pool:
                        for batch in batches:
                            futures = []
                            for strategy in batch:
                                future = pool.submit(
                                    StrategyMutatorV2.evaluate_strategy_wrapper,
                                    (strategy, market_data)
                                )
                                futures.append((strategy, future))
                            
                            for strategy, future in futures:
                                try:
                                    result = ray.get(future)
                                    strategy.oos_metrics = result
                                    logger.info(f"Evaluated {strategy.name}: "
                                              f"score={strategy.score():.2f}, "
                                              f"trades={result.get('trade_count', 0)}")
                                except Exception as e:
                                    logger.error(f"Failed to evaluate {strategy.name}: {str(e)}")
                                    strategy.oos_metrics = self._default_metrics()
                else:
                    logger.info("Using local sequential backtesting")
                    for strategy in new_strategies:
                        try:
                            result = StrategyMutatorV2.evaluate_strategy_wrapper((strategy, market_data))
                            strategy.oos_metrics = result
                            logger.info(f"Evaluated {strategy.name}: "
                                      f"score={strategy.score():.2f}, "
                                      f"trades={result.get('trade_count', 0)}")
                        except Exception as e:
                            logger.error(f"Failed to evaluate {strategy.name}: {str(e)}")
                            strategy.oos_metrics = self._default_metrics()
                
                self.backtest_time = time.time() - backtest_start

            # Step 4: Sort and select top strategies
            self.strategy_pool.sort(key=lambda s: s.score(), reverse=True)
            
            if not self.strategy_pool:
                logger.warning("Strategy pool empty, reinitializing")
                self._reinitialize_pool()
                return
                
            # Calculate RL usage in the current pool
            rl_count = sum(1 for s in self.strategy_pool if 'rl_agent' in s.features)
            self.rl_usage = rl_count / len(self.strategy_pool)
            self.config.rl_usage = self.rl_usage  # Pass to config for mutation adaptation

            # --- IMMUNITY: Save top 10% strategies unchanged ---
            immune_count = int(0.1 * len(self.strategy_pool))
            immune_strategies = self.strategy_pool[:immune_count]
            
            # --- TOURNAMENT SELECTION for the rest ---
            tournament_size = 5
            selected = []
            top_k = min(self.config.top_k, len(self.strategy_pool))
            # We need to select (top_k - immune_count) strategies
            for _ in range(top_k - immune_count):
                # Randomly select tournament_size candidates
                candidates = random.sample(self.strategy_pool, tournament_size)
                # Select the best from the candidates
                winner = max(candidates, key=lambda s: s.score())
                selected.append(winner)
            
            # Combine immune and tournament selected
            self.strategy_pool = immune_strategies + selected
            # Sort again and keep only top_k
            self.strategy_pool.sort(key=lambda s: s.score(), reverse=True)
            self.strategy_pool = self.strategy_pool[:top_k]

            # Calculate feature importance
            try:
                method = "SHAP" if self.use_shap else "Default"
                logger.info(f"Calculating feature importance using {method} method...")
                self.importance = self.importance_calculator.compute(self.strategy_pool)
                
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
                self.importance = DefaultFeatureImportanceCalculator().compute(self.strategy_pool)

            self.config.top_strategies = self.strategy_pool

            # Step 5: Generate mutants
            new = []
            self.operator_usage = []
            
            for _ in range(self.config.n_mutants):
                if random.random() < self.config.mutation_probs["crossover"] and len(self.strategy_pool) >= 2:
                    p1, p2 = random.sample(self.strategy_pool, 2)
                    parent_feats = {**p1.features, **p2.features}
                    parent_name = f"{p1.name}+{p2.name}"
                else:
                    weights = [max(s.score(), 0.01) for s in self.strategy_pool]
                    parent = random.choices(self.strategy_pool, weights=weights)[0]
                    parent_feats = dict(parent.features)
                    parent_name = parent.name

                mutant_feats, op_name = self._tweak(parent_feats)
                name = f"mut_{uuid.uuid4().hex[:8]}"
                new_strat = StrategyEmbedding(name=name, features=mutant_feats)
                new.append(new_strat)
                
                self.pending_evaluation[name] = (op_name, parent_name)
                self.operator_usage.append((op_name, name))

            self.strategy_pool.extend(new)

            # Step 6: Adapt mutation probabilities
            self.config.adapt_mutation_probs()

            logger.info("\nOperator Statistics:")
            for op, stats in self.config.operator_stats.items():
                if stats['n'] > 0:
                    avg_impact = stats['total_impact'] / stats['n']
                    prob = self.config.mutation_probs.get(op, 0.0)
                    logger.info(f"  {op}: n={stats['n']} avg_impact={avg_impact:.3f} prob={prob:.2f}")

            if new:
                best_new_score = max(s.score() for s in new) if new else 0
                self.config.update_based_on_performance(best_new_score)

            best = self.strategy_pool[0] if self.strategy_pool else None
            logger.info(f"[EVO] Gen {self.step}: pool={len(self.strategy_pool)} "
                  f"best={best.name if best else 'N/A'} score={best.score() if best else 0:.2f}")

            self._tick += 1
            if self._tick % self.checkpoint_every == 0:
                self.save_checkpoint()
                
            # Compress all strategies
            for s in self.strategy_pool:
                s.compress()
                
        except Exception as e:
            logger.error(f"Evolution error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self._reinitialize_pool()

    # CRITICAL FIX: Improved pool reinitialization
    def _reinitialize_pool(self):
        logger.info("[mutator] Reinitializing strategy pool with enhanced strategies")
        self.strategy_pool = [
            StrategyEmbedding.create_random(self.feature_bank)
            for _ in range(50)
        ]
        
        # Reset pending evaluations
        self.pending_evaluation = {}
        
        # Reset operator statistics
        self.config.operator_stats = {op: {"n": 0, "total_impact": 0} 
                                    for op in self.config.operator_stats}
        
        # Boost mutation probabilities
        for op in self.config.mutation_probs:
            self.config.mutation_probs[op] = min(0.8, self.config.mutation_probs[op] * 1.3)
    
    @staticmethod
    def evaluate_strategy_wrapper(args: Tuple[StrategyEmbedding, pd.DataFrame]) -> Dict[str, float]:
        strategy, market_data = args
        try:
            from mutator_evo.backtest.backtrader_adapter import BacktraderAdapter
            data_feed = bt.feeds.PandasData(dataname=market_data)
            adapter = BacktraderAdapter(data_feed)
            return adapter.evaluate(strategy)
        except Exception as e:
            import traceback
            logger = logging.getLogger(__name__)
            logger.error(f"Backtest failed for {strategy.name}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "oos_sharpe": -1.0,  # Less severe penalty
                "oos_max_drawdown": 50.0,
                "oos_win_rate": 0.3,
                "overfitting_penalty": 0.8,
                "trade_count": 10
            }
    
    # CRITICAL FIX: Better default values
    def _default_metrics(self):
        return {
            "oos_sharpe": -1.0,  # Less severe penalty
            "oos_max_drawdown": 50.0,
            "oos_win_rate": 0.3,
            "overfitting_penalty": 0.8,
            "trade_count": 10
        }

    def save_checkpoint(self, path: Optional[str] = None) -> None:
        try:
            import dill
        except ImportError:
            import pickle as dill
            
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"mutator_checkpoint_{ts}.pkl"
        path = self.checkpoint_dir / filename if path is None else pathlib.Path(path)
        
        try:
            for s in self.strategy_pool + self.archival_pool:
                s.compress()
                
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
                    "rl_usage": self.rl_usage
                }, f)
            logger.info(f"Checkpoint saved: {path}")
            
            if self.max_checkpoints > 0:
                checkpoint_files = sorted(
                    self.checkpoint_dir.glob("mutator_checkpoint_*.pkl"),
                    key=lambda f: f.stat().st_mtime,
                    reverse=True
                )
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
            self.rl_usage = state.get("rl_usage", 0.0)
            
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