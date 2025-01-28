import pandas as pd
from execution.helpers.multiplier_fitting import fit_multipliers

def test_fit_multipliers():
    """Test that fit_multipliers runs successfully."""
    example_data = pd.DataFrame({
        "Exercise": ["Bench Press"],
        "Week #": [1],
        "Set #": [3],
        "# of Reps": [5],
        "Multiplier of Max": [1.2]
    })
    
    result = fit_multipliers(example_data)
    assert isinstance(result, dict), "❌ Expected output to be a dictionary!"
