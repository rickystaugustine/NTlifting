import pandas as pd
from execution.helpers.weight_assignment import assign_weights

def test_assign_weights():
    """Test the assign_weights function to ensure it properly calculates assigned weights."""

    merged_data = pd.DataFrame({
        "Exercise": ["Bench Press", "Squat"],
        "Relevant Core": ["Bench", "Squat"],
        "Multiplier of Max": [0.75, 0.85],
        "Player": ["John Doe", "Jane Doe"],
        "Tested Max": [200, 250]  # ✅ Added column to prevent KeyError
    })

    flattened_core_maxes_df = pd.DataFrame({
        "Player": ["John Doe", "Jane Doe"],
        "Relevant Core": ["Bench", "Squat"],
        "Bench Press": [200, 180],
        "Squat": [300, 250]
    })

    exercise_functions = {"Bench Press": "Bench", "Squat": "Squat"}

    df = assign_weights(merged_data, flattened_core_maxes_df, exercise_functions)
    assert not df.empty, "❌ assign_weights() returned an empty DataFrame!"
