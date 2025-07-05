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
from collections import deque, defaultdict
from datetime import date
from typing import Any, Dict, List, Set, Optional, Tuple
import pandas as pd
import backtrader as bt
import zlib
import hashlib

from .config import DynamicConfig
from .ray_pool import RayPool
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

class StrategyEmbedding:
    def __init__(self, name: str, features: Dict[str, Any]):
        self.name = name
        self._features = features
        self._compressed: Optional[bytes] = None
        self.oos_metrics: Optional[Dict[str, float]] = None
        self._hash: Optional[str] = None

    @property
    def features(self) -> Dict[str, Any]:
        if self._compressed is not None:
            self._decompress()
        return self._features

    @features.setter
    def features(self, value: Dict[str, Any]):
        self._features = value
        self._compressed = None
        self._hash = None

    def compress(self):
        if self._compressed is None and self._features is not None:
            json_str = json.dumps(self._features, sort_keys=True)
            self._compressed = zlib.compress(json_str.encode(), level=3)
            del self._features
            self._features = None

    def _decompress(self):
        if self._compressed is not None:
            json_str = zlib.decompress(self._compressed).decode()
            self._features = json.loads(json_str)
            self._compressed = None

    def decompress(self):
        self._decompress()

    def get_hash(self) -> str:
        if self._hash is None:
            if self._compressed is not None:
                self._decompress()
            features_str = json.dumps(self._features, sort_keys=True)
            self._hash = hashlib.sha256(features_str.encode()).hexdigest()
        return self._hash

    def score(self) -> float:
        if not self.oos_metrics:
            return -10.0
        
        sharpe = self.oos_metrics.get("oos_sharpe", 0)
        dd = self.oos_metrics.get("oos_max_drawdown", 100)
        penalty = self.oos_metrics.get("overfitting_penalty", 1.0)
        
        trade_count = self.oos_metrics.get("trade_count", 0)
        if trade_count < 10:
            return -5.0
        
        return (sharpe * 0.7 - dd * 0.3) * (1 - penalty)

    @classmethod
    def create_random(cls, feature_bank: Set[str]) -> "StrategyEmbedding":
        name = f"rand_{uuid.uuid4().hex[:8]}"
        features = {}
        
        num_features = min(random.randint(3, 8), len(feature_bank))
        selected_features = random.sample(list(feature_bank), num_features)
        
        for feat in selected_features:
            if feat in {"trade_size", "stop_loss", "take_profit"}:
                if feat == "trade_size":
                    features[feat] = random.uniform(0.1, 0.3)
                elif feat == "stop_loss":
                    features[feat] = random.uniform(0.01, 0.05)
                elif feat == "take_profit":
                    features[feat] = random.uniform(0.03, 0.08)
            elif "period" in feat or "window" in feat:
                features[feat] = random.randint(5, 50)
            elif feat == "rl_agent":
                features[feat] = {
                    "hidden_layers": [
                        random.randint(32, 128) 
                        for _ in range(random.randint(1, 3))
                    ],
                    "learning_rate": 10**random.uniform(-4, -2),
                    "gamma": random.uniform(0.8, 0.99),
                    "epsilon": random.uniform(0.05, 0.2)
                }
            else:
                features[feat] = random.random() > 0.3
                
        strategy = cls(name, features)
        strategy.compress()
        return strategy

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
        
        self.operator_usage = []
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

        if not self.strategy_pool:
            self._reinitialize_pool()
            return

        self.step += 1
        
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

            # Step 4: Sort and select top strategies
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
                self.importance = DefaultFeatureImportanceCalculator().compute(top)

            self.strategy_pool = top.copy()
            self.config.top_strategies = top

            # Step 5: Generate mutants
            new = []
            self.operator_usage = []
            
            for _ in range(self.config.n_mutants):
                if random.random() < self.config.mutation_probs["crossover"] and len(top) >= 2:
                    p1, p2 = random.sample(top, 2)
                    parent_feats = {**p1.features, **p2.features}
                    parent_name = f"{p1.name}+{p2.name}"
                else:
                    weights = [max(s.score(), 0.01) for s in top]
                    parent = random.choices(top, weights=weights)[0]
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

    def _reinitialize_pool(self):
        logger.info("[mutator] Reinitializing strategy pool with 50 strategies")
        self.strategy_pool = [
            StrategyEmbedding.create_random(self.feature_bank)
            for _ in range(50)
        ]
        self.pending_evaluation = {}
    
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
                "oos_sharpe": -5.0,
                "oos_max_drawdown": 100.0,
                "oos_win_rate": 0.0,
                "overfitting_penalty": 1.0,
                "trade_count": 0
            }
    
    def _default_metrics(self):
        return {
            "oos_sharpe": -5.0,
            "oos_max_drawdown": 100.0,
            "oos_win_rate": 0.0,
            "overfitting_penalty": 1.0,
            "trade_count": 0
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