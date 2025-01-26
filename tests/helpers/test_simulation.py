# tests/helpers/test_simulation.py
import pytest
from execution.helpers.simulation import run_simulation

def test_run_simulation():
    """Ensure simulation runs without errors."""
    example_input = {"athlete": "John Doe", "max_lift": 200}
    result = run_simulation(example_input)
    assert isinstance(result, dict)  # Adjust based on expected output

