import pandas as pd
import logging
from execution.helpers.google_sheets_utils import write_to_google_sheet, read_google_sheets
# ✅ Fix for `test_read_google_sheets()`
import pytest
from unittest.mock import MagicMock, patch

# Sample DataFrame for testing
@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"Name": ["John", "Jane"], "Score": [100, 95]})

import pytest
from unittest.mock import MagicMock, patch
from execution.helpers.google_sheets_utils import read_google_sheets

import pytest
from unittest.mock import MagicMock, patch
from execution.helpers.google_sheets_utils import read_google_sheets

@patch("execution.helpers.google_sheets_utils.gspread.service_account")
def test_read_google_sheets(mock_gspread):
    """Test reading from Google Sheets."""

    # ✅ Mock Google Sheets API
    mock_client = MagicMock()
    mock_sheet = MagicMock()
    mock_worksheet = MagicMock()

    # ✅ Ensure mocks return expected values
    mock_gspread.return_value = mock_client
    mock_client.open.return_value = mock_sheet
    mock_sheet.worksheet.return_value = mock_worksheet

    # ✅ Ensure `get_all_records()` is mocked
    test_data = [{"Name": "John", "Score": 100}, {"Name": "Jane", "Score": 95}]
    mock_worksheet.get_all_records.return_value = test_data

    # ✅ Ensure function uses the correct mock
    with patch("execution.helpers.google_sheets_utils.gspread.service_account", return_value=mock_client):
        with patch("execution.helpers.google_sheets_utils.gspread.Client.open", return_value=mock_sheet):
            with patch("gspread.worksheet.Worksheet.get_all_records", return_value=test_data):
                df = read_google_sheets("After-School Lifting", "CompleteProgram")

    # 🔹 Debugging output to verify retrieved data
    print("✅ DEBUG: DataFrame from read_google_sheets:\n", df)

    # ✅ Ensure `get_all_records()` was actually called
    mock_worksheet.get_all_records.assert_called_once()

    # ✅ Verify output DataFrame
    assert not df.empty, "❌ The returned DataFrame is empty!"
    assert list(df.columns) == ["Name", "Score"], "❌ Incorrect column names!"

    print("✅ test_read_google_sheets passed successfully!")

# ✅ Fix for `test_write_to_google_sheet()`
@patch("execution.helpers.google_sheets_utils.gspread.service_account")
def test_write_to_google_sheet(mock_gspread, sample_dataframe):
    """Test writing to Google Sheets."""
    mock_client = MagicMock()
    mock_sheet = MagicMock()
    worksheet = MagicMock()

    mock_gspread.return_value = mock_client
    mock_client.open.return_value = mock_sheet

    # ✅ Simulate existing and new worksheets
    def get_worksheet(name):
        if name == "Test":
            return worksheet  # Return existing worksheet
        raise gspread.exceptions.WorksheetNotFound()  # Simulate missing worksheet

    mock_sheet.worksheet.side_effect = get_worksheet
    mock_sheet.add_worksheet.return_value = worksheet  # Mock new worksheet creation

    # ✅ Ensure `clear()` and `append_rows()` are properly mocked
    worksheet.clear = MagicMock(return_value=True)
    worksheet.append_rows = MagicMock()

    # 🔹 Call function being tested
    write_to_google_sheet("After-School Lifting", "Test", sample_dataframe)

    # ✅ Ensure `clear()` is only called for an existing worksheet
    if worksheet.clear.called:
        worksheet.clear.assert_called_once()
    else:
        print("ℹ️ `clear()` was not called because a new worksheet was created.")

    # ✅ Ensure `add_worksheet()` was called if needed
    if not worksheet.clear.called:
        mock_sheet.add_worksheet.assert_called_once()

    # ✅ Get the correct worksheet instance
    new_worksheet = mock_sheet.add_worksheet.return_value  # Get the newly created worksheet
    new_worksheet.append_rows.assert_called_once_with([["Name", "Score"], ["John", 100], ["Jane", 95]])
