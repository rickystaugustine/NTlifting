import pandas as pd
import pytest
from execution.helpers.data_processing import preprocess_data

def test_merge_data():
    """Test if `preprocess_data` correctly merges program and core maxes data."""

    # 🔹 Mock example data for `program_df`
    program_df = pd.DataFrame({
        "Exercise": ["Bench", "Squat"],
        "Code": [1, 2],
        "Week #": [1, 1],
        "Set #": [3, 3],
        "# of Reps": [5, 5],
        "Multiplier of Max": [0.75, 0.85],
        "Relevant Core": ["Bench", "Barbell Squat"],  # Ensure this column is present
        "Player": ["John Doe", "Jane Doe"]
    })

    # 🔹 Mock example data for `core_maxes_df`
    core_maxes_df = pd.DataFrame({
        "Player": ["John Doe", "Jane Doe"],
        "Relevant Core": ["Bench", "Barbell Squat"],  # Ensure this column is present
        "Bench": [200, 180],
        "Barbell Squat": [300, 250],
        "Clean": [150, 130],
        "Hex-Bar Deadlift": [350, 320]
    })

    # ✅ Debugging Step: Print Columns Before Processing
    print("🔍 core_maxes_df Columns:", core_maxes_df.columns.tolist())

    # ✅ Debugging Step: Print Data Before Processing
    print("🔍 core_maxes_df Data:\n", core_maxes_df)

    # 🔹 Call `preprocess_data`
    flattened_core_maxes_df, repeated_program_df = preprocess_data(program_df, core_maxes_df)

    # ✅ Verify `Relevant Core` is still in both DataFrames
    assert "Relevant Core" in flattened_core_maxes_df.columns, "❌ 'Relevant Core' missing from core maxes DataFrame!"
    assert "Relevant Core" in repeated_program_df.columns, "❌ 'Relevant Core' missing from program DataFrame!"

    # ✅ Verify `Player` is in both DataFrames
    assert "Player" in flattened_core_maxes_df.columns, "❌ 'Player' missing from core maxes DataFrame!"
    assert "Player" in repeated_program_df.columns, "❌ 'Player' missing from program DataFrame!"

    # ✅ Check if merge was successful (merged values should not be NaN)
    merged_data = repeated_program_df.merge(
        flattened_core_maxes_df, 
        on=["Player", "Relevant Core"], 
        how="left", 
        suffixes=("", "_core")  # Rename the second occurrence
    )

    # Ensure `Tested Max` is correctly assigned
    # Explicitly set dtype before filling NaN values to prevent future downcasting issues
    merged_data["Tested Max"] = merged_data["Tested Max"].astype(float).fillna(merged_data["Tested Max_core"]).astype(float)

    merged_data.drop(columns=["Tested Max_core"], inplace=True)
    assert not merged_data["Tested Max"].isna().all(), "❌ Merge failed: All Tested Max values are missing!"

    print("✅ Merge test passed successfully!")
