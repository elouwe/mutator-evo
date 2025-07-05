# src/tests/benchmark/test_ray_performance.py
import pytest
import time
import numpy as np
import backtrader as bt
from mutator_evo.core.strategy_mutator import StrategyMutatorV2
from mutator_evo.backtest.backtrader_adapter import BacktraderAdapter
from mutator_evo.core.strategy_embedding import StrategyEmbedding
import pandas as pd

@pytest.fixture(scope="module")
def market_data_fixture():
    # Generate synthetic market data for testing
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': np.random.uniform(90, 110, 100),
        'high': np.random.uniform(110, 120, 100),
        'low': np.random.uniform(80, 90, 100),
        'close': np.random.uniform(95, 105, 100),
        'volume': np.random.randint(10000, 50000, 100)
    }, index=dates)
    return bt.feeds.PandasData(dataname=data)

@pytest.mark.benchmark
def test_ray_scaling(market_data_fixture):
    """Test scalability using Ray distributed computing."""
    # Initialize backtesting adapter with market data
    adapter = BacktraderAdapter(market_data_fixture)
    results = {}
    
    # Without Ray (sequential execution)
    mutator = StrategyMutatorV2(backtest_adapter=adapter, use_ray=False)
    # Create initial strategy pool
    mutator.strategy_pool = [StrategyEmbedding.create_random({'param1', 'param2'}) for _ in range(10)]
    
    # Measure execution time without Ray
    start = time.time()
    mutator.evolve()
    results["no_ray"] = time.time() - start
    
    # With Ray (if available)
    try:
        import ray
        mutator.use_ray = True
        # Measure execution time with Ray
        start = time.time()
        mutator.evolve()
        results["with_ray"] = time.time() - start
        
        # Calculate and report speedup
        speedup = results["no_ray"] / results["with_ray"]
        print(f"\nRay speedup: {speedup:.2f}x")
    except ImportError:
        pytest.skip("Ray not available")
    
    # Verify baseline test completed
    assert results["no_ray"] > 0, "No-Ray test failed"