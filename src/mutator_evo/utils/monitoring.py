# src/mutator_evo/utils/monitoring.py
from prometheus_client import start_http_server, Gauge, Counter, Summary
import time
import logging

logger = logging.getLogger(__name__)

class EvolutionMonitor:
    """Class for monitoring evolution progress with Prometheus metrics"""
    def __init__(self, port=8000):
        self.port = port
        self._init_metrics()
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")
    
    def _init_metrics(self):
        # Gauges
        self.generation = Gauge('evolution_generation', 'Current generation number')
        self.best_score = Gauge('evolution_best_score', 'Best strategy score in current generation')
        self.avg_score = Gauge('evolution_avg_score', 'Average strategy score in current generation')
        self.pool_size = Gauge('evolution_pool_size', 'Number of strategies in the pool')
        self.feature_count = Gauge('evolution_feature_count', 'Average feature count per strategy')
        self.rl_usage = Gauge('evolution_rl_usage', 'Percentage of strategies using RL agent')
        
        # Counters
        self.mutation_ops = Counter('evolution_mutation_operations', 'Count of mutation operations', ['operator'])
        self.errors = Counter('evolution_errors', 'Count of errors', ['type'])
        self.slack_alerts = Counter('evolution_slack_alerts', 'Count of Slack alerts sent')
        
        # Summaries
        self.backtest_time = Summary('evolution_backtest_time', 'Time spent on backtesting')
        self.mutation_time = Summary('evolution_mutation_time', 'Time spent on mutation operations')
    
    def update_generation_metrics(self, generation, best_score, avg_score, pool_size, feature_count, rl_usage):
        self.generation.set(generation)
        self.best_score.set(best_score)
        self.avg_score.set(avg_score)
        self.pool_size.set(pool_size)
        self.feature_count.set(feature_count)
        self.rl_usage.set(rl_usage)
    
    def count_mutation(self, operator_name):
        self.mutation_ops.labels(operator=operator_name).inc()
    
    def count_error(self, error_type):
        self.errors.labels(type=error_type).inc()
    
    def record_backtest_time(self, duration):
        self.backtest_time.observe(duration)
    
    def record_mutation_time(self, duration):
        self.mutation_time.observe(duration)