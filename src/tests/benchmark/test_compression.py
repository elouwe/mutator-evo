# src/tests/benchmark/test_compression.py

import pytest
import random
import numpy as np
from mutator_evo.core.strategy_embedding import StrategyEmbedding
import pickle
import sys
import json
from collections.abc import Iterable

def deep_compare(a, b, path="", verbose=True):
    """Recursive comparison of structures with detailed difference output"""
    if type(a) != type(b):
        if verbose: 
            print(f"Type mismatch at {path}: {type(a)} vs {type(b)}")
        return False
        
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            if verbose:
                missing = set(a.keys()) - set(b.keys())
                extra = set(b.keys()) - set(a.keys())
                print(f"Key mismatch at {path}: missing {missing}, extra {extra}")
            return False
            
        for key in a:
            new_path = f"{path}.{key}" if path else key
            if not deep_compare(a[key], b[key], new_path, verbose):
                return False
        return True
        
    elif isinstance(a, list):
        if len(a) != len(b):
            if verbose: 
                print(f"List length mismatch at {path}: {len(a)} vs {len(b)}")
            return False
            
        for i, (x, y) in enumerate(zip(a, b)):
            new_path = f"{path}[{i}]"
            if not deep_compare(x, y, new_path, verbose):
                return False
        return True
        
    elif isinstance(a, np.ndarray):
        if a.shape != b.shape:
            if verbose: 
                print(f"Array shape mismatch at {path}: {a.shape} vs {b.shape}")
            return False
            
        if not np.array_equal(a, b):
            if verbose:
                print(f"Array values differ at {path}")
                # Show first 5 differences
                diff_indices = np.argwhere(a != b)
                for idx in diff_indices[:5]:
                    idx_str = ",".join(map(str, idx))
                    print(f"  At index {idx_str}: {a[tuple(idx)]} vs {b[tuple(idx)]}")
            return False
        return True
        
    elif isinstance(a, float):
        # Allow small tolerance for float comparison
        if not np.isclose(a, b, atol=1e-6):
            if verbose:
                print(f"Float value mismatch at {path}: {a} vs {b}")
            return False
        return True
        
    else:
        if a != b:
            if verbose:
                print(f"Value mismatch at {path}: {a} vs {b}")
            return False
        return True

@pytest.mark.benchmark
def test_compression_ratio():
    """Test compression ratio for strategies with numpy data."""
    # Create test features with random values
    features = {
        f'feature_{i}': float(random.random()) for i in range(100)
    }
    # Add RL agent weights as numpy arrays
    features.update({
        'rl_agent': {
            'hidden_layers': [random.randint(32, 128) for _ in range(3)],
            'weights': [np.random.randn(64, 64).tolist() for _ in range(2)]
        }
    })
    
    # Save a copy for comparison
    original_features = json.loads(json.dumps(features))
    
    strat = StrategyEmbedding("compression_test", features)
    
    # Measure original size
    pickled_data = pickle.dumps(strat.features, protocol=pickle.HIGHEST_PROTOCOL)
    uncompressed_size = len(pickled_data)
    
    # Compress and measure compressed size
    strat.compress()
    compressed_size = len(strat._compressed)
    
    # Calculate compression ratio
    ratio = uncompressed_size / compressed_size
    print(f"\nCompression: {uncompressed_size} -> {compressed_size} bytes (ratio: {ratio:.2f}x)")
    
    # Verify minimum compression ratio
    assert ratio > 1.5, f"Compression ratio too low: {ratio:.2f}x"
    
    # Decompress and compare with original
    decompressed = strat.features
    
    # Verify data integrity
    assert deep_compare(original_features, decompressed), "Decompressed data doesn't match original"