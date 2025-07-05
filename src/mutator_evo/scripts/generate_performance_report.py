# src/mutator_evo/scripts/generate_performance_report.py
import json
import matplotlib.pyplot as plt
import os
from datetime import datetime
from mutator_evo.core.strategy_embedding import StrategyEmbedding
from mutator_evo.backtest.backtrader_adapter import BacktraderAdapter

def generate_performance_report(mutator, adapter, output_dir="reports"):
    """Generate a comprehensive performance report"""
    # Create reports directory
    os.makedirs(output_dir, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "compression": StrategyEmbedding.get_compression_stats(),
        "cache": {
            "hits": adapter._cache_hits,
            "misses": adapter._cache_misses,
            "hit_rate": adapter._cache_hits / (adapter._cache_hits + adapter._cache_misses) 
                       if (adapter._cache_hits + adapter._cache_misses) > 0 else 0
        },
        "generations": mutator.performance_metrics.get("generations", 0),
        "timings": {
            "avg_backtest": mutator.performance_metrics.get("backtest_time", 0) / 
                            max(1, mutator.performance_metrics.get("generations", 1)),
            "avg_mutation": mutator.performance_metrics.get("mutation_time", 0) / 
                            max(1, mutator.performance_metrics.get("generations", 1))
        }
    }
    
    # Save JSON report
    json_path = os.path.join(output_dir, "performance_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # Generate charts
    if report["generations"] > 0:
        plt.figure(figsize=(10, 6))
        plt.bar(["Backtest", "Mutation"], 
                [report["timings"]["avg_backtest"], report["timings"]["avg_mutation"]])
        plt.title("Average Operation Times")
        plt.ylabel("Seconds")
        plt.savefig(os.path.join(output_dir, "timings_chart.png"))
    
    print(f"Performance report generated in {output_dir}")
    return report

if __name__ == "__main__":
    # Example usage (in real code, pass actual objects)
    from mutator_evo.core.strategy_mutator import StrategyMutatorV2
    from mutator_evo.backtest.backtrader_adapter import BacktraderAdapter
    
    # Here should be real initialized objects
    mutator = StrategyMutatorV2()
    adapter = BacktraderAdapter()
    
    generate_performance_report(mutator, adapter)