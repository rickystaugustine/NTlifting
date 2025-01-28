import pandas as pd
import pytest
from execution.helpers.data_processing import preprocess_data

def test_preprocess_data():
    """Test the preprocess_data function with mock input."""

    # 🔹 Mock `program_df`
    program_df = pd.DataFrame({
        "Exercise": ["Bench Press", "Squat"],
        "Sets": [3, 4],
        "Relevant Core": ["Bench", "Barbell Squat"],  # ✅ Ensure this column is present
        "Player": ["John Doe", "Jane Smith"]
    })

    # 🔹 Mock `core_maxes_df`
    core_maxes_df = pd.DataFrame({
        "Player": ["John Doe", "Jane Smith"],
        "Relevant Core": ["Bench", "Barbell Squat"],  # ✅ Ensure this column is present
        "Bench": [200, 150],
        "Barbell Squat": [300, 250],
        "Clean": [180, 140],
        "Hex-Bar Deadlift": [350, 280]
    })

    # 🔹 Call `preprocess_data`
    flattened_core_maxes, repeated_program = preprocess_data(program_df, core_maxes_df)

    # ✅ Verify `"Relevant Core"` is still in both DataFrames
    assert "Relevant Core" in flattened_core_maxes.columns, "❌ 'Relevant Core' missing from core maxes DataFrame!"
    assert "Relevant Core" in repeated_program.columns, "❌ 'Relevant Core' missing from program DataFrame!"

    # ✅ Verify `"Player"` is in both DataFrames
    assert "Player" in flattened_core_maxes.columns, "❌ 'Player' missing from core maxes DataFrame!"
    assert "Player" in repeated_program.columns, "❌ 'Player' missing from program DataFrame!"

    print("✅ preprocess_data test passed successfully!")
