# tests/helpers/test_multiplier_fitting.py
import pytest
from execution.helpers.multiplier_fitting import fit_multipliers

def test_fit_multipliers():
    """Test that fit_multipliers runs successfully."""
    example_data = {"bench_press": [100, 120, 140]}  # Replace with valid test data
    result = fit_multipliers(example_data)
    assert isinstance(result, dict)  # Adjust based on actual return type

