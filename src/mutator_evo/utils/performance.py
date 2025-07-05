# src/mutator_evo/utils/performance.py

import time
import ray
from functools import wraps
import logging

logger = logging.getLogger(__name__)

def track_performance(func):
    """Decorator for tracking function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        
        # Collect metrics
        metrics = {
            "function": func.__name__,
            "duration": duration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ray_available": ray.is_initialized(),
        }
        
        if ray.is_initialized():
            metrics["ray_resources"] = ray.available_resources()
        
        # Log metrics
        logger.info(f"Performance metrics: {metrics}")
        return result
    return wrapper