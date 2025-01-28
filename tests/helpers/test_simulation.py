import pandas as pd
from execution.helpers.simulation import run_simulation

def test_run_simulation():
    """Ensure simulation runs without errors."""
    example_input = pd.DataFrame({
        "Player": ["John Doe"],
        "Exercise": ["Bench Press"],
        "Assigned Weight": [200],
        "# of Reps": [5]
    })

    result = run_simulation(example_input)
    assert result is not None, "❌ Simulation returned None!"
