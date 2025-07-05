# src/mutator_evo/core/strategy_embedding.py
import zlib
import json
import uuid
import random
import sys
import pickle
import numpy as np
from typing import Dict, Any, Optional, Set
import hashlib

class StrategyEmbedding:
    """Class for representing and compressing trading strategies."""
    
    _compression_stats = {
        'compress_count': 0,
        'total_original_size': 0,
        'total_compressed_size': 0
    }

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

    def _optimize_rl_weights(self, weights: list) -> list:
        """Optimizes RL agent weights for better compression."""
        optimized = []
        for weight_matrix in weights:
            if isinstance(weight_matrix, np.ndarray):
                # Already optimized
                optimized.append(weight_matrix)
            elif isinstance(weight_matrix, list):
                # Convert to numpy array
                try:
                    arr = np.array(weight_matrix, dtype=np.float32)
                    optimized.append(arr)
                except:
                    optimized.append(weight_matrix)
            else:
                optimized.append(weight_matrix)
        return optimized

    def compress(self):
        """Compresses strategy features with handling of numpy types and large lists."""
        if self._compressed is None and self._features is not None:
            try:
                # Optimize RL agent weights before compression
                if 'rl_agent' in self._features:
                    rl_agent = self._features['rl_agent']
                    if isinstance(rl_agent, dict) and 'weights' in rl_agent:
                        rl_agent['weights'] = self._optimize_rl_weights(rl_agent['weights'])
                
                # Measure original size
                pickled_data = pickle.dumps(self._features, protocol=pickle.HIGHEST_PROTOCOL)
                original_size = len(pickled_data)
                
                # Convert numpy arrays and large lists to binary format
                def convert_for_compression(obj):
                    # Handle numpy arrays
                    if isinstance(obj, np.ndarray):
                        # Use float32 to save space
                        if obj.dtype == np.float64:
                            obj = obj.astype(np.float32)
                        return {
                            '__numpy__': True,
                            'data': obj.tobytes(),
                            'dtype': str(obj.dtype),
                            'shape': obj.shape
                        }
                    # Handle large numerical lists (like RL agent weights)
                    elif isinstance(obj, list) and len(obj) > 10:
                        try:
                            # Convert to numpy for efficient compression
                            arr = np.array(obj, dtype=np.float32)
                            if arr.dtype.kind in 'iuf':  # integer, unsigned, float
                                return {
                                    '__numpy__': True,
                                    'data': arr.tobytes(),
                                    'dtype': str(arr.dtype),
                                    'shape': arr.shape
                                }
                        except:
                            pass  # Leave as-is if conversion fails
                    # Recursively process dictionaries
                    elif isinstance(obj, dict):
                        return {k: convert_for_compression(v) for k, v in obj.items()}
                    # Recursively process lists
                    elif isinstance(obj, list):
                        return [convert_for_compression(x) for x in obj]
                    return obj
                    
                processed_features = convert_for_compression(self._features)
                
                # Use maximum compression level (9)
                compressed_data = zlib.compress(
                    pickle.dumps(processed_features, protocol=pickle.HIGHEST_PROTOCOL), 
                    level=9
                )
                
                # Store compressed data
                self._compressed = compressed_data
                
                # Update compression statistics
                StrategyEmbedding._compression_stats['compress_count'] += 1
                StrategyEmbedding._compression_stats['total_original_size'] += original_size
                StrategyEmbedding._compression_stats['total_compressed_size'] += len(compressed_data)
                
                # Free memory
                del self._features
                self._features = None
            except Exception as e:
                raise ValueError(f"Compression failed: {str(e)}") from e

    def _decompress(self):
        if self._compressed is not None:
            try:
                # First decompress
                pickled_data = zlib.decompress(self._compressed)
                
                # Then deserialize
                processed_features = pickle.loads(pickled_data)
                
                # Restore numpy arrays
                def restore_from_compression(obj):
                    if isinstance(obj, dict) and '__numpy__' in obj:
                        # Restore numpy array
                        arr = np.frombuffer(
                            obj['data'],
                            dtype=obj['dtype']
                        ).reshape(obj['shape'])
                        return arr
                    elif isinstance(obj, dict):
                        return {k: restore_from_compression(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [restore_from_compression(x) for x in obj]
                    return obj
                    
                decompressed_features = restore_from_compression(processed_features)
                
                # Convert RL agent weights back to lists
                if 'rl_agent' in decompressed_features:
                    rl_agent = decompressed_features['rl_agent']
                    if isinstance(rl_agent, dict) and 'weights' in rl_agent:
                        weights = rl_agent['weights']
                        if isinstance(weights, list):
                            # Convert numpy arrays back to lists
                            rl_agent['weights'] = [
                                w.tolist() if isinstance(w, np.ndarray) else w 
                                for w in weights
                            ]
                
                # Convert all numpy arrays to lists for consistency
                def convert_numpy_to_list(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_to_list(x) for x in obj]
                    return obj
                
                self._features = convert_numpy_to_list(decompressed_features)
                self._compressed = None
            except Exception as e:
                raise ValueError(f"Decompression failed: {str(e)}") from e

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
                    "epsilon": random.uniform(0.05, 0.2),
                    # Use float32 instead of float64 to save space
                    "weights": [np.random.randn(64, 64).astype(np.float32).tolist() for _ in range(2)]
                }
            else:
                features[feat] = random.random() > 0.3
                
        strategy = cls(name, features)
        strategy.compress()
        return strategy

    @classmethod
    def get_compression_stats(cls):
        if cls._compression_stats['compress_count'] == 0:
            return None
            
        return {
            "average_ratio": cls._compression_stats['total_original_size'] / cls._compression_stats['total_compressed_size'],
            "total_original": cls._compression_stats['total_original_size'],
            "total_compressed": cls._compression_stats['total_compressed_size'],
            "count": cls._compression_stats['compress_count']
        }