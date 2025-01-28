import gspread
import pandas as pd
import logging
import os
from oauth2client.service_account import ServiceAccountCredentials
from gspread.exceptions import SpreadsheetNotFound, WorksheetNotFound

# ✅ Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - INFO - %(message)s")

# ✅ Google Sheets Authorization
def authorize_google_client():
    """Authorize and return Google Sheets client."""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    
    credentials_path = "/Users/ricky.staugustine/.config/gspread/service_account.json"

    if not os.path.exists(credentials_path):
        logging.error(f"❌ ERROR: Google Sheets credentials file missing at {credentials_path}.")
        return None  # Prevent failure

    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    client = gspread.authorize(creds)
    logging.info("✅ Google Sheets API authorization successful!")
    return client
    
def read_google_sheets(spreadsheet_name, worksheet_name):
    """Reads data from a specified Google Sheets spreadsheet and worksheet."""
    client = authorize_google_client()
    if client is None:
        logging.error("❌ ERROR: Google Sheets authentication failed.")
        return pd.DataFrame()  # Return an empty DataFrame to prevent failure

    try:
        sheet = client.open(spreadsheet_name)
    except SpreadsheetNotFound:
        logging.error(f"❌ ERROR: Spreadsheet '{spreadsheet_name}' not found.")
        return pd.DataFrame()

    try:
        worksheet = sheet.worksheet(worksheet_name)
    except WorksheetNotFound:
        logging.error(f"❌ ERROR: Worksheet '{worksheet_name}' not found in '{spreadsheet_name}'.")
        return pd.DataFrame()

    # ✅ Debugging output before calling `get_all_records()`
    logging.info(f"✅ DEBUG: Calling get_all_records() for '{spreadsheet_name}' -> '{worksheet_name}'")

    # ✅ **Ensure `get_all_records()` is actually called**
    data = worksheet.get_all_records()

    # ✅ Debugging: Check retrieved data
    logging.info(f"✅ DEBUG: Retrieved {len(data)} records from '{spreadsheet_name}' -> '{worksheet_name}'")

    if not data:
        logging.warning(f"⚠️ WARNING: No data found in '{spreadsheet_name}' -> '{worksheet_name}'. Returning empty DataFrame.")
        return pd.DataFrame()

    return pd.DataFrame(data)

# ✅ Write data to Google Sheets
def write_to_google_sheet(spreadsheet_name, worksheet_name, data):
    """Writes data to Google Sheets."""
    client = authorize_google_client()
    if client is None:
        return None  # Prevent failure if credentials are missing

    try:
        sheet = client.open(spreadsheet_name)
    except SpreadsheetNotFound:
        logging.error(f"❌ ERROR: Spreadsheet '{spreadsheet_name}' not found. Creating a new one.")
        sheet = client.create(spreadsheet_name)

    try:
        worksheet = sheet.worksheet(worksheet_name)
    except WorksheetNotFound:
        logging.warning(f"⚠️ Worksheet '{worksheet_name}' not found in '{spreadsheet_name}'. Creating a new one.")
        worksheet = sheet.add_worksheet(title=worksheet_name, rows="1000", cols="26")

    # Convert DataFrame to list if needed
    if isinstance(data, pd.DataFrame):
        data = [data.columns.tolist()] + data.values.tolist()

    print("🔍 DEBUG: Type of worksheet.clear ->", type(worksheet.clear))
    cleared = worksheet.clear()
    logging.info(f"✅ Worksheet cleared: {cleared}")

    worksheet.append_rows(data)
    logging.info(f"✅ Successfully wrote data to {spreadsheet_name}/{worksheet_name}.")

