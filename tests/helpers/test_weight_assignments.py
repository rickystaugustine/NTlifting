import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from execution.helpers.weight_assignment import assign_weights

@patch("execution.helpers.weight_assignment.write_to_google_sheet")
def test_assign_weights(mock_write_to_google_sheet):
    """Test the assign_weights function to ensure it properly calculates assigned weights."""
    
    # Run the function
    df = assign_weights()

    # ✅ Ensure function output is a DataFrame
    assert isinstance(df, pd.DataFrame), "❌ Output should be a DataFrame"

    # ✅ Ensure essential columns exist
    required_columns = {"Player", "Exercise", "Assigned Weight", "Relevant Core", "Fitted Multiplier"}
    missing_cols = required_columns - set(df.columns)
    assert not missing_cols, f"❌ Missing expected columns: {missing_cols}"

    # ✅ Ensure no missing Assigned Weights
    assert df["Assigned Weight"].isna().sum() == 0, "❌ 'Assigned Weight' column contains NaN values!"

    # ✅ Ensure 'Assigned Weight' has expected data types
    assert df["Assigned Weight"].dtype == object, "❌ 'Assigned Weight' should allow mixed types (numerics + 'NRM')!"

    # ✅ Ensure "NRM" values are correctly assigned
    assert "NRM" in df["Assigned Weight"].values, "❌ 'NRM' is missing from Assigned Weights!"

    # ✅ Ensure `write_to_google_sheet` was called
    mock_write_to_google_sheet.assert_called_once_with("After-School Lifting", "AssignedWeights", df)

    print("✅ All assign_weights tests passed!")

if __name__ == "__main__":
    pytest.main()
