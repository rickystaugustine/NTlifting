# tests/helpers/test_simulation.py
import pytest
from execution.helpers.simulation import run_simulation

def test_run_simulation():
    """Ensure simulation runs without errors."""
    example_input = {
        "athlete": "John Doe",
        "max_lift": 200,
        "# of Reps": 5  # ✅ Ensure this column is included
    }
    result = run_simulation(example_input)

    assert result is not None
    assert isinstance(result, dict) and "simulated_data" in result
