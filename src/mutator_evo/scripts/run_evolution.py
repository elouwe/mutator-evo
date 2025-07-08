# src/mutator_evo/scripts/run_evolution.py
import sys
import os
import random
import pandas as pd
import numpy as np
import backtrader as bt
import traceback
import logging
import datetime
import json
import time
from pathlib import Path
import ray

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mutator_evo.core.strategy_mutator import StrategyMutatorV2
from mutator_evo.core.strategy_embedding import StrategyEmbedding
from mutator_evo.backtest.backtrader_adapter import BacktraderAdapter
from mutator_evo.utils.memory_tracker import MemoryTracker
from mutator_evo.utils.monitoring import EvolutionMonitor
from mutator_evo.utils.alerts import AlertManager
from mutator_evo.scripts.generate_performance_report import generate_performance_report
from mutator_evo.scripts.visualize_evolution import plot_evolution

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"evolution_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def load_market_data() -> bt.feeds.PandasData:
    # Try to use real market data
    try:
        import yfinance as yf
        logger.info("Downloading real market data from Yahoo Finance...")
        # Download S&P 500 index data
        df = yf.download('^GSPC', start='2020-01-01', end='2023-01-01')
        # Rename columns to match expected structure
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        # Remove rows with missing values
        df.dropna(inplace=True)
        return bt.feeds.PandasData(dataname=df)
    except Exception as e:
        logger.error(f"Failed to download real data: {str(e)}. Using synthetic data...")
        
        # Generate synthetic data as fallback
        logger.info("Generating realistic market data with strong trends and volatility...")
        dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
        n = len(dates)
        np.random.seed(42)
        
        prices = np.ones(n) * 100
        trend_direction = 1
        
        trend_lengths = [60, 90, 120, 75, 100]
        trend_idx = 0
        
        for i in range(0, n, trend_lengths[trend_idx % len(trend_lengths)]):
            trend_length = trend_lengths[trend_idx % len(trend_lengths)]
            trend_idx += 1
            
            if i + trend_length < n:
                trend_strength = 0.03 + 0.05 * random.random()
                
                for j in range(i, i + int(trend_length * 0.8)):
                    if j < n:
                        prices[j] = prices[j-1] * (1 + trend_direction * trend_strength / trend_length)
                
                for j in range(i + int(trend_length * 0.8), i + trend_length):
                    if j < n:
                        prices[j] = prices[j-1] * (1 - trend_direction * trend_strength / (trend_length * 0.5))
                
                trend_direction *= -1
        
        for i in range(1, n):
            volatility = 0.015 * (1 + 0.8 * np.sin(i/20))
            change = np.random.normal(0, volatility)
            prices[i] = prices[i] * (1 + change)
            prices[i] = max(0.1, prices[i])
        
        opens = np.zeros(n)
        highs = np.zeros(n)
        lows = np.zeros(n)
        volumes = np.zeros(n)
        
        opens[0] = prices[0]
        for i in range(1, n):
            opens[i] = prices[i-1]
            
            intraday_vol = 0.015 * (1 + 0.8 * np.sin(i/20))
            change = np.random.normal(0, intraday_vol)
            prices[i] = prices[i] * (1 + change)
            
            daily_range = 0.015 * (1 + 1.0 * random.random())
            high_ratio = 1 + daily_range * random.random()
            low_ratio = 1 - daily_range * random.random()
            
            highs[i] = max(prices[i] * high_ratio, prices[i])
            lows[i] = min(prices[i] * low_ratio, prices[i])
            
            volumes[i] = 20000 + int(100000 * abs(change) / intraday_vol)
        
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        }, index=dates)
        
        return bt.feeds.PandasData(dataname=df)

def initialize_mutator(data_feed) -> StrategyMutatorV2:
    adapter = BacktraderAdapter(full_data_feed=data_feed)
    return StrategyMutatorV2(
        backtest_adapter=adapter,
        use_shap=True,
        top_k=15,
        n_mutants=15,
        max_age=8,
        checkpoint_every=15,
        max_checkpoints=10,
        mutation_probs={
            "add": 0.55,
            "drop": 0.45,
            "shift": 0.40,
            "invert": 0.30,
            "metaboost": 0.25,
            "crossover": 0.60,
            "uniform_crossover": 0.45,
            "rl_mutation": 0.35,
        },
        checkpoint_dir="checkpoints"
    )

def initialize_feature_bank(mutator: StrategyMutatorV2):
    mutator.feature_bank = {
        'use_ema', 'use_sma', 'use_rsi', 'use_macd', 'use_stoch',
        'use_bollinger', 'use_obv', 'use_adx',
        'ema_period', 'sma_period', 'rsi_period', 
        'macd_fast', 'macd_slow', 'macd_signal',
        'stoch_k', 'stoch_d', 'bollinger_period',
        'adx_period', 'trade_size', 'stop_loss', 'take_profit',
        'rl_agent'
    }

def run_evolution(mutator: StrategyMutatorV2, monitor: EvolutionMonitor, 
                 alert_manager: AlertManager, epochs: int = 100):
    logger.info("Starting evolution...")
    best_scores = []
    degradation_count = 0
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        logger.info(f"\nEpoch {epoch}/{epochs}")
        try:
            # Add new random strategies every 20 epochs for diversity
            if epoch % 20 == 0:
                logger.info("Adding 10 new random strategies for diversity...")
                new_strategies = [StrategyEmbedding.create_random(mutator.feature_bank) for _ in range(10)]
                mutator.strategy_pool.extend(new_strategies)
            
            # Emergency measures: if average score < -3 for two consecutive generations, reset pool
            if len(best_scores) >= 2 and best_scores[-1] < -3 and best_scores[-2] < -3:
                logger.error("Critical degradation detected! Resetting pool...")
                mutator.strategy_pool = [
                    StrategyEmbedding.create_random(mutator.feature_bank)
                    for _ in range(50)
                ]
                best_scores = []
                degradation_count = 0
            
            if epoch == 30:
                mutator.config._params["mutation_probs"] = {
                    "add": 0.65,
                    "drop": 0.55,
                    "shift": 0.50,
                    "invert": 0.40,
                    "metaboost": 0.35,
                    "crossover": 0.70,
                    "uniform_crossover": 0.55,
                    "rl_mutation": 0.45,
                }
                logger.info("Increased mutation probabilities for more exploration")
            
            mutator.evolve()
            
            # Update metrics
            if mutator.strategy_pool:
                best = max(mutator.strategy_pool, key=lambda s: s.score())
                best_score = best.score()
                best_scores.append(best_score)
                
                # Calculate metrics for monitoring
                pool_size = len(mutator.strategy_pool)
                avg_score = sum(s.score() for s in mutator.strategy_pool) / pool_size
                feature_count = sum(len(s.features) for s in mutator.strategy_pool) / pool_size
                rl_usage = sum(1 for s in mutator.strategy_pool if 'rl_agent' in s.features) / pool_size
                
                # Update Prometheus metrics
                monitor.update_generation_metrics(
                    generation=epoch,
                    best_score=best_score,
                    avg_score=avg_score,
                    pool_size=pool_size,
                    feature_count=feature_count,
                    rl_usage=rl_usage
                )
                
                # Track mutation operators
                if hasattr(mutator, 'operator_usage'):
                    for operator, _ in mutator.operator_usage:
                        monitor.count_mutation(operator)
                
                # Check for degradation
                if len(best_scores) > 3:
                    if best_scores[-1] < best_scores[-2] < best_scores[-3]:
                        degradation_count += 1
                        if degradation_count >= 3:
                            alert_manager.send_degradation_alert(epoch, best_scores[-5:])
                            degradation_count = 0  # Reset after alert
                    else:
                        degradation_count = 0
                
                # Record timing metrics
                epoch_time = time.time() - epoch_start
                monitor.record_mutation_time(epoch_time)
                
                # Record backtest time if available
                if hasattr(mutator, 'backtest_time'):
                    monitor.record_backtest_time(mutator.backtest_time)
            
        except Exception as e:
            logger.error(f"Epoch error: {str(e)}")
            logger.error(traceback.format_exc())
            monitor.count_error('epoch_error')
            mutator.strategy_pool = [
                StrategyEmbedding.create_random(mutator.feature_bank)
                for _ in range(50)
            ]
            mutator.pending_evaluation = {}
            
            try:
                mutator.save_checkpoint(f"emergency_checkpoint_epoch{epoch}.pkl")
                logger.info("Saved emergency checkpoint")
            except Exception as e:
                logger.error(f"Failed to save emergency checkpoint: {str(e)}")

def generate_visualizations():
    try:
        logger.info("Generating visualizations...")
        
        vis_dir = Path("visualizations")
        vis_dir.mkdir(exist_ok=True)
        
        plot_evolution("checkpoints")
        logger.info("Visualizations generated")
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        logger.error(traceback.format_exc())

def main():
    ray_initialized = False
    use_ray = True
    
    # Start memory tracker
    mem_tracker = MemoryTracker(interval=300)  # Snapshot every 5 minutes
    mem_tracker.start()
    logger.info("Memory tracking started")
    
    # Initialize monitoring and alerting
    monitor = EvolutionMonitor(port=8000)
    alert_manager = AlertManager()
    
    try:
        try:
            import ray
            from ray._private import services
            logger.info(f"Ray version: {ray.__version__}")
            logger.info("Ray is available, will use distributed computing")
        except ImportError:
            logger.warning("Ray not installed, falling back to local execution")
            use_ray = False
            
        if use_ray:
            init_args = {
                "num_cpus": 4,
                "ignore_reinit_error": True,
                "logging_level": logging.WARNING,
                "object_store_memory": 2 * 10**9,
                "_system_config": {
                    "max_direct_call_object_size": 10**6,
                    "port_retries": 100,
                    "gcs_server_request_timeout_seconds": 30,
                    "gcs_rpc_server_reconnect_timeout_s": 30,
                },
                "include_dashboard": False,
                "dashboard_host": "127.0.0.1",
                "dashboard_port": None
            }
            
            try:
                ray.init(**init_args)
                ray_initialized = True
                logger.info("Ray initialized for distributed computing")
            except Exception as e:
                logger.error(f"Ray initialization failed: {str(e)}")
                logger.info("Falling back to local execution")
                use_ray = False
        else:
            logger.info("Using local execution without Ray")
        
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        os.chdir(results_dir)
        
        logger.info("Loading market data...")
        data_feed = load_market_data()
        
        logger.info("Initializing mutator...")
        mutator = initialize_mutator(data_feed)
        initialize_feature_bank(mutator)
        mutator.use_ray = use_ray
        
        logger.info("Creating initial pool of 30 random strategies...")
        mutator.strategy_pool = [
            StrategyEmbedding.create_random(mutator.feature_bank)
            for _ in range(30)
        ]
        
        logger.info("Starting evolution process...")
        run_evolution(mutator, monitor, alert_manager, epochs=100)
        
        logger.info("\nEvolution completed successfully!")
        if mutator.strategy_pool:
            best = max(mutator.strategy_pool, key=lambda s: s.score())
            best.decompress()
            
            logger.info(f"Best strategy: {best.name}")
            logger.info(f"Features: {list(best.features.keys())}")
            logger.info(f"Score: {best.score():.2f}")
            
            if best.oos_metrics:
                logger.info("Metrics details:")
                for metric, value in best.oos_metrics.items():
                    logger.info(f"  {metric}: {value}")
            
            with open("best_strategy.txt", "w") as f:
                f.write(f"Name: {best.name}\n")
                f.write(f"Score: {best.score():.2f}\n")
                f.write(f"OOS Sharpe: {best.oos_metrics.get('oos_sharpe', 0):.2f}\n")
                f.write(f"OOS Drawdown: {best.oos_metrics.get('oos_max_drawdown', 0):.2f}%\n")
                f.write(f"OOS Win Rate: {best.oos_metrics.get('oos_win_rate', 0):.2f}\n")
                f.write(f"Trade count: {best.oos_metrics.get('trade_count', 0)}\n")
                f.write("\nFeatures:\n")
                for k, v in best.features.items():
                    f.write(f"  {k}: {v}\n")
            
            with open("best_strategy.json", "w") as f:
                json.dump({
                    "name": best.name,
                    "features": best.features,
                    "score": best.score(),
                    "metrics": best.oos_metrics
                }, f, indent=2)
        else:
            logger.warning("No strategies in pool after evolution")
        
        generate_visualizations()
        logger.info("All operations completed successfully!")
    except Exception as e:
        logger.critical(f"Fatal error in main process: {str(e)}")
        logger.critical(traceback.format_exc())
        monitor.count_error('fatal_error')
        try:
            generate_visualizations()
        except:
            logger.error("Failed to generate visualizations after error")
    finally:
        if ray_initialized:
            ray.shutdown()
            logger.info("Ray resources released")
        
        # Stop memory tracker and generate report
        mem_tracker.stop()
        report_path = "memory_usage_report.txt"
        mem_tracker.generate_report(report_path)
        logger.info(f"Memory report generated: {report_path}")
        
        # Generate performance report
        if 'mutator' in locals() and 'adapter' in locals():
            perf_report = generate_performance_report(mutator, adapter)
            if perf_report:
                logger.info(f"Performance report generated with {len(perf_report)} records")
            else:
                logger.warning("Performance report generation returned empty")
        else:
            logger.warning("Skipping performance report because mutator or adapter is not defined")
        
        logger.info("Program terminated")

if __name__ == "__main__":
    main()