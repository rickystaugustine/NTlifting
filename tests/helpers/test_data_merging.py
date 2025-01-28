import pytest
import pandas as pd
from execution.helpers.data_processing import preprocess_data  # Ensure import is correct

def test_merge_data():
    """ Test if merge_data correctly combines program and core maxes data. """
    
    # 🔹 Mock example data for `program_df`
    program_df = pd.DataFrame({
        "Exercise": ["Bench", "Squat"],
        "Code": [1, 2],
        "# of Reps": [5, 5]
    })

    # 🔹 Mock example data for `core_maxes_df`
    core_maxes_df = pd.DataFrame({
        "Player": ["John Doe", "Jane Doe"],
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

    # ✅ Debugging Step: Print Processed Data
    print("✅ Processed Flattened Core Maxes:\n", flattened_core_maxes_df)
    print("✅ Processed Repeated Program:\n", repeated_program_df)

    # 🔹 Ensure `flattened_core_maxes_df` is not empty
    assert not flattened_core_maxes_df.empty, "❌ ERROR: Flattened core maxes dataframe is empty!"
    assert not repeated_program_df.empty, "❌ ERROR: Repeated program dataframe is empty!"
