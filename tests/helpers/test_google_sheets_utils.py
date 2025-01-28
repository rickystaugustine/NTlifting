import pytest
import pandas as pd
import logging
from unittest.mock import patch, MagicMock
from execution.helpers.google_sheets_utils import write_to_google_sheet, read_google_sheets

# Sample DataFrame for testing
@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"Name": ["John", "Jane"], "Score": [100, 95]})

# ✅ Fix for `test_read_google_sheets()`
@patch("execution.helpers.google_sheets_utils.gspread.service_account")
def test_read_google_sheets(mock_gspread):
    """Test reading from Google Sheets."""
    mock_client = MagicMock()
    mock_sheet = MagicMock()
    mock_worksheet = MagicMock()

    mock_gspread.return_value = mock_client
    mock_client.open.return_value = mock_sheet
    mock_sheet.worksheet.return_value = mock_worksheet

    # ✅ **Fix: Ensure `get_all_records()` is a function returning `test_data`**
    test_data = [{"Name": "John", "Score": 100}, {"Name": "Jane", "Score": 95}]
    mock_worksheet.get_all_records = MagicMock(return_value=test_data)

    df = read_google_sheets("Test Sheet", "Scores")

    # ✅ Ensure `get_all_records()` was actually called
    mock_worksheet.get_all_records.assert_called_once()

    # ✅ Assertions
    assert isinstance(df, pd.DataFrame), "Returned object is not a DataFrame"
    assert not df.empty, "DataFrame is empty"
    assert list(df.columns) == ["Name", "Score"], "Column names do not match expected values"
    assert df.iloc[0]["Name"] == "John", "First row 'Name' does not match"
    assert df.iloc[1]["Score"] == 95, "Second row 'Score' does not match"

# ✅ Fix for `test_write_to_google_sheet()`
@patch("execution.helpers.google_sheets_utils.gspread.service_account")
def test_write_to_google_sheet(mock_gspread, sample_dataframe):
    """Test writing to Google Sheets."""
    mock_client = MagicMock()
    mock_sheet = MagicMock()
    worksheet = MagicMock()

    mock_gspread.return_value = mock_client
    mock_client.open.return_value = mock_sheet
    mock_sheet.worksheet.return_value = worksheet

    # ✅ **Fix: Ensure `clear()` is properly mocked**
    worksheet.clear = MagicMock()

    write_to_google_sheet("Test Sheet", "Scores", sample_dataframe)

    # ✅ Ensure `clear()` was called once
    worksheet.clear.assert_called_once()
    logging.info("✅ DEBUG: worksheet.clear() was successfully called once!")

    # ✅ Ensure `append_rows()` was called with the correct structure
    expected_data = [["Name", "Score"], ["John", 100], ["Jane", 95]]
    worksheet.append_rows.assert_called_once_with(expected_data)
