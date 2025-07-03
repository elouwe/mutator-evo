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
from pathlib import Path

# Добавим путь к корню проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Теперь импортируем наши модули
from mutator_evo.core.strategy_mutator import StrategyMutatorV2
from mutator_evo.core.strategy_embedding import StrategyEmbedding
from mutator_evo.backtest.backtrader_adapter import BacktraderAdapter

# Настройка логирования
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
    logger.info("Generating realistic market data with strong trends and volatility...")
    dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
    n = len(dates)
    np.random.seed(42)
    
    # Create strong upward and downward trends
    prices = np.ones(n) * 100
    trend_direction = 1
    
    # Different trend lengths for more realism
    trend_lengths = [60, 90, 120, 75, 100]
    trend_idx = 0
    
    for i in range(0, n, trend_lengths[trend_idx % len(trend_lengths)]):
        trend_length = trend_lengths[trend_idx % len(trend_lengths)]
        trend_idx += 1
        
        if i + trend_length < n:
            # Stronger trends (3-8%)
            trend_strength = 0.03 + 0.05 * random.random()
            
            # Trend phase (80% of the period)
            for j in range(i, i + int(trend_length * 0.8)):
                if j < n:
                    prices[j] = prices[j-1] * (1 + trend_direction * trend_strength / trend_length)
            
            # Correction phase (20% of the period)
            for j in range(i + int(trend_length * 0.8), i + trend_length):
                if j < n:
                    prices[j] = prices[j-1] * (1 - trend_direction * trend_strength / (trend_length * 0.5))
            
            # Reverse direction
            trend_direction *= -1
    
    # Add volatility
    for i in range(1, n):
        # Higher volatility (1.5% base + sine wave modulation)
        volatility = 0.015 * (1 + 0.8 * np.sin(i/20))
        change = np.random.normal(0, volatility)
        prices[i] = prices[i] * (1 + change)
        
        # Ensure prices don't go negative
        prices[i] = max(0.1, prices[i])
    
    # Generate OHLCV data
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
        
        # Larger daily ranges (1.5-3%)
        daily_range = 0.015 * (1 + 1.0 * random.random())
        high_ratio = 1 + daily_range * random.random()
        low_ratio = 1 - daily_range * random.random()
        
        highs[i] = max(prices[i] * high_ratio, prices[i])
        lows[i] = min(prices[i] * low_ratio, prices[i])
        
        # Volume proportional to volatility
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
        use_shap=True,  # Включить SHAP-based feature importance
        top_k=15,  # Keep more top strategies
        n_mutants=15,  # Generate more mutants
        max_age=8,  # Faster rotation
        checkpoint_every=15,
        max_checkpoints=10,
        mutation_probs={
            "add": 0.55,
            "drop": 0.45,
            "shift": 0.40,
            "invert": 0.30,
            "metaboost": 0.25,
            "crossover": 0.60,
            "uniform_crossover": 0.45,  # New
            "rl_mutation": 0.35,        # New
        },
        checkpoint_dir="checkpoints"  # Explicit checkpoint directory
    )

def initialize_feature_bank(mutator: StrategyMutatorV2):
    mutator.feature_bank = {
        'use_ema', 'use_sma', 'use_rsi', 'use_macd', 'use_stoch',
        'use_bollinger', 'use_obv', 'use_adx',
        'ema_period', 'sma_period', 'rsi_period', 
        'macd_fast', 'macd_slow', 'macd_signal',
        'stoch_k', 'stoch_d', 'bollinger_period',
        'adx_period', 'trade_size', 'stop_loss', 'take_profit',
        'rl_agent'  # New
    }

def run_evolution(mutator: StrategyMutatorV2, epochs: int = 100):
    logger.info("Starting evolution...")
    for epoch in range(1, epochs + 1):
        logger.info(f"\nEpoch {epoch}/{epochs}")
        try:
            # Dynamically increase mutation probabilities after 30 epochs
            if epoch == 30:
                mutator.config._params["mutation_probs"] = {
                    "add": 0.65,
                    "drop": 0.55,
                    "shift": 0.50,
                    "invert": 0.40,
                    "metaboost": 0.35,
                    "crossover": 0.70,
                    "uniform_crossover": 0.55,  # New
                    "rl_mutation": 0.45,        # New
                }
                logger.info("Increased mutation probabilities for more exploration")
            
            mutator.evolve()
        except Exception as e:
            logger.error(f"Epoch error: {str(e)}")
            logger.error(traceback.format_exc())
            # Reset pool with more strategies
            mutator.strategy_pool = [
                StrategyEmbedding.create_random(mutator.feature_bank)
                for _ in range(25)
            ]
            mutator.pending_evaluation = {}
            
            # Save emergency checkpoint
            try:
                mutator.save_checkpoint(f"emergency_checkpoint_epoch{epoch}.pkl")
                logger.info("Saved emergency checkpoint")
            except Exception as e:
                logger.error(f"Failed to save emergency checkpoint: {str(e)}")

def generate_visualizations():
    try:
        from mutator_evo.scripts.visualize_evolution import plot_evolution
        logger.info("Generating visualizations...")
        
        # Create visualizations directory
        vis_dir = Path("visualizations")
        vis_dir.mkdir(exist_ok=True)
        
        plot_evolution("checkpoints")
        logger.info("Visualizations generated")
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        logger.error(traceback.format_exc())

def main():
    try:
        # Create results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        os.chdir(results_dir)
        
        logger.info("Loading market data...")
        data_feed = load_market_data()
        
        logger.info("Initializing mutator...")
        mutator = initialize_mutator(data_feed)
        initialize_feature_bank(mutator)
        
        logger.info("Creating initial pool of 30 random strategies...")
        mutator.strategy_pool = [
            StrategyEmbedding.create_random(mutator.feature_bank)
            for _ in range(30)
        ]
        
        logger.info("Starting evolution process...")
        run_evolution(mutator, epochs=100)
        
        logger.info("\nEvolution completed successfully!")
        if mutator.strategy_pool:
            best = max(mutator.strategy_pool, key=lambda s: s.score())
            logger.info(f"Best strategy: {best.name}")
            logger.info(f"Features: {list(best.features.keys())}")
            logger.info(f"Score: {best.score():.2f}")
            if best.oos_metrics:
                logger.info(f"OOS Sharpe: {best.oos_metrics.get('oos_sharpe', 0):.2f}")
                logger.info(f"OOS Drawdown: {best.oos_metrics.get('oos_max_drawdown', 0):.2f}%")
                logger.info(f"OOS Win Rate: {best.oos_metrics.get('oos_win_rate', 0):.2f}")
                logger.info(f"Trade count: {best.oos_metrics.get('trade_count', 0)}")
                
                # Save best strategy details
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
        else:
            logger.warning("No strategies in pool after evolution")
        
        # Generate visualizations after evolution
        generate_visualizations()
        
        logger.info("All operations completed successfully!")
    except Exception as e:
        logger.critical(f"Fatal error in main process: {str(e)}")
        logger.critical(traceback.format_exc())
        try:
            # Try to generate visualizations even on error
            generate_visualizations()
        except:
            logger.error("Failed to generate visualizations after error")
        finally:
            logger.info("Program terminated due to error")

if __name__ == "__main__":
    main()