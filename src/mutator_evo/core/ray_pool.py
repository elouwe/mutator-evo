# src/mutator_evo/core/ray_pool.py
import ray
from contextlib import contextmanager
from typing import List, Callable, Any, Iterator
import logging
import os

logger = logging.getLogger(__name__)

class RayPool:
    def __init__(self, num_cpus: int = None):
        self.num_cpus = num_cpus
        self.initialized_externally = ray.is_initialized()
        
    @contextmanager
    def pool(self) -> Iterator['RayPool']:
        """Context manager for managing the Ray pool."""
        try:
            # Reuse existing connection if Ray is already initialized
            if ray.is_initialized():
                logger.info("Ray already initialized, reusing existing connection")
                yield self
                return
                
            # Parameters to avoid port conflicts
            init_args = {
                "num_cpus": self.num_cpus,
                "ignore_reinit_error": True,
                "logging_level": logging.WARNING,
                "_system_config": {
                    "max_direct_call_object_size": 10**6,
                    "port_retries": 100,  # Increase attempts to find available port
                    "gcs_server_request_timeout_seconds": 30,  # Increase timeout
                    "gcs_rpc_server_reconnect_timeout_s": 30,  # Reconnect timeout
                },
                "include_dashboard": False,  # Disable dashboard
                "dashboard_host": "127.0.0.1",  # Use local host
                "dashboard_port": None  # No fixed port
            }
            
            # Initialize Ray
            try:
                ray.init(**init_args)
                logger.info(f"Ray initialized with {self.num_cpus or 'default'} CPUs")
            except Exception as e:
                logger.error(f"Ray initialization failed: {str(e)}")
                # Try alternative method
                logger.info("Attempting alternative Ray initialization method")
                try:
                    ray.init(address="auto", **init_args)
                    logger.info("Ray initialized with 'auto' address")
                except Exception as e2:
                    logger.error(f"Alternative Ray initialization failed: {str(e2)}")
                    raise
                    
            yield self
        finally:
            # Shutdown only if we initialized Ray
            if not self.initialized_externally and ray.is_initialized():
                ray.shutdown()
                logger.info("Ray resources released")

    def map(self, func: Callable, items: List[Any]) -> List[Any]:
        """Parallel execution of a function on a list of items."""
        if not ray.is_initialized():
            raise RuntimeError("Ray is not initialized. Use within 'with RayPool() as pool' context")
        
        # Handle nested calls by running sequentially
        if self.initialized_externally:
            logger.warning("Nested Ray call detected, running sequentially")
            return [func(item) for item in items]
        
        remote_func = ray.remote(func)
        futures = [remote_func.remote(item) for item in items]
        return ray.get(futures)
    
    def submit(self, func: Callable, item: Any) -> ray.ObjectRef:
        """Submit a task for execution and return a future."""
        if not ray.is_initialized():
            raise RuntimeError("Ray is not initialized. Use within 'with RayPool() as pool' context")
        
        # Handle nested calls by executing immediately
        if self.initialized_externally:
            logger.warning("Nested Ray call detected, running sequentially")
            class ImmediateResult:
                def __init__(self, result):
                    self.result = result
                def __call__(self):
                    return self.result
            return ImmediateResult(func(item))
        
        remote_func = ray.remote(func)
        return remote_func.remote(item)