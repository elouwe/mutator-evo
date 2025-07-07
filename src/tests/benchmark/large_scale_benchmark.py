import pytest
import time
import numpy as np
import logging
import sys
import random
from tqdm import tqdm
import os
import psutil
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed

# Create results directory if not exists
os.makedirs("results/tests", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("results/tests/benchmark.log")  # Updated path
    ]
)
logger = logging.getLogger(__name__)

# Global flag for verbose output
VERBOSE = False

class ProgressBar:
    def __init__(self, total, desc="", unit="it"):
        self.pbar = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            file=sys.stdout,
            disable=not VERBOSE,
            mininterval=1
        )
        self.start_time = time.time()
        
    def update(self, n=1):
        self.pbar.update(n)
        
    def close(self):
        self.pbar.close()

def generate_strategy_batch(n: int) -> list:
    """Generate n strategies efficiently"""
    feature_bank = [
        'use_ema', 'use_sma', 'use_rsi', 'use_macd', 'use_stoch',
        'use_bollinger', 'use_obv', 'use_adx',
        'ema_period', 'sma_period', 'rsi_period', 
        'macd_fast', 'macd_slow', 'macd_signal',
        'stoch_k', 'stoch_d', 'bollinger_period',
        'adx_period', 'trade_size', 'stop_loss', 'take_profit'
    ]
    
    logger.info(f"Generating {n} strategies...")
    pbar = ProgressBar(n, "Generating strategies")
    strategies = []
    
    for i in range(n):
        strategy = {
            'name': f"strat_{i}",
            'features': {
                k: random.random() for k in random.sample(feature_bank, random.randint(3, 8))
            }
        }
        strategies.append(strategy)
        pbar.update()
    
    pbar.close()
    return strategies

def evaluate_strategy_local(strategy, market_data):
    """Efficient strategy evaluation"""
    complexity = len(strategy['features'])
    processing_time = 0.01 * complexity + random.uniform(0, 0.05)
    time.sleep(processing_time)
    
    feature_count = len(strategy['features'])
    score = sum(strategy['features'].values()) / feature_count
    
    sharpe = score * 2 - 1 + random.gauss(0, 0.1)
    drawdown = (1 - score) * 50 + random.gauss(0, 5)
    win_rate = 0.4 + score * 0.3 + random.gauss(0, 0.05)
    
    return {
        "sharpe": max(-5, min(5, sharpe)),
        "max_drawdown": max(0, min(100, drawdown)),
        "win_rate": max(0, min(1, win_rate)),
        "trade_count": int(50 + score * 150 + random.gauss(0, 20)),
        "score": score,
        "processing_time": processing_time,
        "strategy_name": strategy['name']
    }

def run_parallel_evaluation(strategies, market_data, num_workers=8):
    """Run evaluation with optimized logging"""
    logger.info(f"Starting evaluation of {len(strategies)} strategies with {num_workers} workers")
    
    start_time = time.time()
    mem_start = psutil.Process(os.getpid()).memory_info().rss
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(evaluate_strategy_local, strategy, market_data): strategy 
                   for strategy in strategies}
        
        results = []
        pbar = ProgressBar(len(strategies), "Evaluating strategies")
        
        for future in as_completed(futures):
            try:
                results.append(future.result())
                pbar.update()
            except Exception as e:
                if VERBOSE:
                    logger.error(f"Evaluation failed: {str(e)}")
                pbar.update()
    
    pbar.close()
    
    total_time = time.time() - start_time
    mem_end = psutil.Process(os.getpid()).memory_info().rss
    
    return results, total_time, mem_end - mem_start

def analyze_results(results):
    """Efficient results analysis with key metrics only"""
    if not results:
        return {}
    
    scores = [r["score"] for r in results]
    sharpes = [r["sharpe"] for r in results]
    processing_times = [r["processing_time"] for r in results]
    
    return {
        "avg_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "min_score": min(scores),
        "avg_sharpe": sum(sharpes) / len(sharpes),
        "avg_processing": sum(processing_times) / len(processing_times),
        "top_strategy": max(results, key=lambda x: x["score"])["strategy_name"]
    }

@pytest.mark.benchmark
def test_10k_strategies_parallel():
    """Benchmark test for 10,000 strategies"""
    global VERBOSE
    VERBOSE = False  # Disable verbose output for benchmark
    
    # Generate market data
    market_data = {
        'open': np.random.rand(1000) * 100 + 50,
        'close': np.random.rand(1000) * 100 + 50,
        'volume': np.random.randint(10000, 50000, 1000)
    }
    
    # Generate strategies
    gen_start = time.time()
    strategies = generate_strategy_batch(10000)
    gen_time = time.time() - gen_start
    
    # Evaluate strategies
    results, eval_time, mem_used = run_parallel_evaluation(strategies, market_data, num_workers=8)
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Print performance report
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY: 10,000 STRATEGIES")
    print("="*50)
    print(f"Strategy Generation: {gen_time:.2f}s")
    print(f"Evaluation Time:     {eval_time:.2f}s")
    print(f"Total Time:          {gen_time + eval_time:.2f}s")
    print(f"Throughput:          {10000/eval_time:.2f} strategies/s")
    print(f"Memory Used:         {mem_used/(1024**2):.2f} MB")
    print("\nKey Statistics:")
    print(f"Average Score:      {analysis['avg_score']:.4f}")
    print(f"Max Strategy Score: {analysis['max_score']:.4f}")
    print(f"Avg Sharpe Ratio:   {analysis['avg_sharpe']:.4f}")
    print(f"Avg Processing:     {analysis['avg_processing']*1000:.2f}ms/strategy")
    print(f"Top Strategy:       {analysis['top_strategy']}")
    print("="*50 + "\n")
    
    # Save benchmark results to file
    with open("results/tests/benchmark_summary.txt", "w") as f:
        f.write("BENCHMARK SUMMARY: 10,000 STRATEGIES\n")
        f.write("="*50 + "\n")
        f.write(f"Strategy Generation: {gen_time:.2f}s\n")
        f.write(f"Evaluation Time:     {eval_time:.2f}s\n")
        f.write(f"Total Time:          {gen_time + eval_time:.2f}s\n")
        f.write(f"Throughput:          {10000/eval_time:.2f} strategies/s\n")
        f.write(f"Memory Used:         {mem_used/(1024**2):.2f} MB\n\n")
        f.write("Key Statistics:\n")
        f.write(f"Average Score:      {analysis['avg_score']:.4f}\n")
        f.write(f"Max Strategy Score: {analysis['max_score']:.4f}\n")
        f.write(f"Avg Sharpe Ratio:   {analysis['avg_sharpe']:.4f}\n")
        f.write(f"Avg Processing:     {analysis['avg_processing']*1000:.2f}ms/strategy\n")
        f.write(f"Top Strategy:       {analysis['top_strategy']}\n")
        f.write("="*50 + "\n")
    
    return {
        "gen_time": gen_time,
        "eval_time": eval_time,
        "throughput": 10000/eval_time,
        "memory_used": mem_used,
        "analysis": analysis
    }

if __name__ == "__main__":
    # For detailed output during development
    VERBOSE = True
    test_10k_strategies_parallel()