import pytest
import pandas as pd
import os
from execution.helpers.simulation import run_simulation

@pytest.fixture(scope="module", autouse=True)
def mock_assigned_weights(tmp_path_factory):
    """Create a mock assigned_weights.pkl file for testing."""
    mock_data = pd.DataFrame({
        "Player": ["John Doe"],
        "Exercise": ["Bench Press"],
        "Assigned Weight": [200],
        "# of Reps": [5]
    })

    data_path = tmp_path_factory.mktemp("data") / "assigned_weights.pkl"
    mock_data.to_pickle(data_path)

    # Override assigned_weights_path in the module
    global assigned_weights_path
    assigned_weights_path = str(data_path)

def test_run_simulation():
    """Ensure simulation runs without errors."""
    example_input = {
        "Player": "John Doe",
        "Exercise": "Bench Press",
        "Assigned Weight": 200,
        "# of Reps": 5
    }
    result = run_simulation(example_input)

    assert result is not None
    assert isinstance(result, dict)
