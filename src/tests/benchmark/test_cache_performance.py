# src/tests/benchmark/test_cache_performance.py
import pytest
import numpy as np
from mutator_evo.backtest.backtrader_adapter import BacktraderAdapter
from mutator_evo.core.strategy_embedding import StrategyEmbedding
import pandas as pd
import backtrader as bt

@pytest.mark.benchmark
def test_cache_hit_rate():
    """Test the effectiveness of backtest caching."""
    # Create minimal market data
    data = pd.DataFrame({
        'open': [100], 'high': [105], 'low': [95], 
        'close': [102], 'volume': [10000]
    })
    adapter = BacktraderAdapter(bt.feeds.PandasData(dataname=data))
    
    # Create test strategy
    strat = StrategyEmbedding("cache_test", {'param': 42})
    
    # First run - cache miss
    result1 = adapter.evaluate(strat)
    assert adapter._cache_hits == 0
    assert adapter._cache_misses == 1
    
    # Second run - cache hit
    result2 = adapter.evaluate(strat)
    assert adapter._cache_hits == 1
    assert result1 == result2
    
    # Calculate and verify hit rate
    hit_rate = adapter._cache_hits / (adapter._cache_hits + adapter._cache_misses)
    print(f"\nCache hit rate: {hit_rate:.1%}")
    assert hit_rate >= 0.5, "Cache hit rate too low"