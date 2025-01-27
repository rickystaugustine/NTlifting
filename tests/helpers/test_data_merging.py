import pytest
import pandas as pd
from execution.helpers.data_merging import merge_data
from execution.helpers.data_processing import preprocess_data

def test_merge_data():
    """Ensure data merging works with valid input."""
    
    # ✅ Mock input with all required columns
    program_mock = pd.DataFrame({
        "Exercise": ["Bench Press", "Squat"],
        "Sets": [3, 4]
    })

    core_maxes_mock = pd.DataFrame({
        "Player": ["John Doe", "Jane Smith"],
        "Bench": [200, 150],
        "Barbell Squat": [300, 250],
        "Clean": [180, 140],
        "Hex-Bar Deadlift": [350, 280]
    })

    # ✅ Ensure preprocess_data() runs correctly
    flattened_core_maxes, repeated_program = preprocess_data(program_mock, core_maxes_mock)

    assert not flattened_core_maxes.empty
    assert not repeated_program.empty
    assert "Relevant Core" in flattened_core_maxes.columns
    assert "Tested Max" in flattened_core_maxes.columns
    assert "Player" in repeated_program.columns
    assert "Exercise" in repeated_program.columns
