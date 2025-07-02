# src/tests/unit/test_mutators.py
import pytest
from mutator_evo.operators.mutation_impl import (
    DropMutation, AddMutation, ShiftMutation, 
    InvertMutation, MetabBoostMutation
)

class MockConfig:
    def __init__(self, probs):
        self.mutation_probs = probs

@pytest.fixture
def mock_config():
    return MockConfig({
        "drop": 1.0,
        "add": 1.0,
        "shift": 1.0,
        "invert": 1.0,
        "metaboost": 1.0
    })

@pytest.fixture
def sample_features():
    return {"num_feat": 10.0, "bool_feat": True, "rare_feat": 5.0}

# DropMutation Tests
def test_drop_mutation_basic(sample_features, mock_config):
    operator = DropMutation()
    result = operator.apply(sample_features, mock_config, set(), {})
    assert len(result) == len(sample_features) - 1
    assert all(k in sample_features for k in result)

def test_drop_mutation_with_importance(sample_features, mock_config):
    operator = DropMutation()
    importance = {"num_feat": 0.9, "bool_feat": 0.1, "rare_feat": 0.5}
    result = operator.apply(sample_features, mock_config, set(), importance)
    assert "bool_feat" not in result  # Должен удалить наименее важную фичу

def test_drop_mutation_disabled():
    config = MockConfig({"drop": 0.0})
    operator = DropMutation()
    features = {"f1": 1.0, "f2": 2.0}
    result = operator.apply(features, config, set(), {})
    assert result == features

def test_drop_mutation_no_common_features(sample_features, mock_config):
    operator = DropMutation()
    # Importance with no common features
    importance = {"unrelated_feat": 0.9}
    result = operator.apply(sample_features, mock_config, set(), importance)
    assert len(result) == len(sample_features) - 1

# AddMutation Tests
def test_add_mutation_basic(sample_features, mock_config):
    operator = AddMutation()
    result = operator.apply(sample_features, mock_config, {"new_feat"}, {})
    assert "new_feat" in result
    assert len(result) == len(sample_features) + 1

def test_add_mutation_with_importance(sample_features, mock_config):
    operator = AddMutation()
    importance = {"new1": 0.9, "new2": 0.1}
    # Run multiple times to ensure probabilistic behavior
    for _ in range(10):
        result = operator.apply(sample_features, mock_config, {"new1", "new2"}, importance)
        assert any(f in result for f in ["new1", "new2"])
        
def test_add_mutation_disabled():
    config = MockConfig({"add": 0.0})
    operator = AddMutation()
    features = {}
    result = operator.apply(features, config, {"new"}, {})
    assert result == features

# ShiftMutation Tests
def test_shift_mutation_basic(sample_features, mock_config):
    operator = ShiftMutation()
    original_value = sample_features["num_feat"]
    result = operator.apply(sample_features, mock_config, set(), {})
    assert result["num_feat"] != original_value
    assert isinstance(result["num_feat"], float)

def test_shift_mutation_non_numeric(sample_features, mock_config):
    operator = ShiftMutation()
    original_value = sample_features["bool_feat"]
    result = operator.apply(sample_features, mock_config, set(), {})
    assert result["bool_feat"] is original_value  # Should remain unchanged

def test_shift_mutation_disabled():
    config = MockConfig({"shift": 0.0})
    operator = ShiftMutation()
    features = {"num": 10.0}
    result = operator.apply(features, config, set(), {})
    assert result == features

# InvertMutation Tests
def test_invert_mutation_basic(sample_features, mock_config):
    operator = InvertMutation()
    original_value = sample_features["bool_feat"]
    result = operator.apply(sample_features, mock_config, set(), {})
    assert result["bool_feat"] != original_value

def test_invert_mutation_non_bool(sample_features, mock_config):
    operator = InvertMutation()
    original_value = sample_features["num_feat"]
    result = operator.apply(sample_features, mock_config, set(), {})
    assert result["num_feat"] == original_value  # Doesn't change the nebulous chips

def test_invert_mutation_disabled():
    config = MockConfig({"invert": 0.0})
    operator = InvertMutation()
    features = {"flag": True}
    result = operator.apply(features, config, set(), {})
    assert result == features

# MetabBoostMutation Tests
def test_metaboost_mutation_basic(sample_features, mock_config):
    operator = MetabBoostMutation()
    result = operator.apply(sample_features, mock_config, set(), {"rare_feat": 0.05})
    assert result["rare_feat"] == 5.0 * 1.2

def test_metaboost_mutation_high_importance(sample_features, mock_config):
    operator = MetabBoostMutation()
    result = operator.apply(sample_features, mock_config, set(), {"num_feat": 0.5})
    assert result["num_feat"] == 10.0  # Doesn't enhance important features

def test_metaboost_mutation_disabled():
    config = MockConfig({"metaboost": 0.0})
    operator = MetabBoostMutation()
    features = {"feat": 5.0}
    result = operator.apply(features, config, set(), {})
    assert result == features