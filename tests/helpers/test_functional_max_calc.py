# tests/helpers/test_functional_max_calc.py
import pytest
from execution.helpers.functional_max_calc import calculate_functional_max

def test_calculate_functional_max():
    """Test functional max calculation logic."""
    example_weights = [100, 120, 140]
    result = calculate_functional_max(example_weights)
    assert isinstance(result, float)  # Adjust based on expected output type

