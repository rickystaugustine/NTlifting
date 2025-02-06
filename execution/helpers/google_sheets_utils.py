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
    
def read_google_sheets(sheet_name, worksheet_name):
    try:
        client = gspread.service_account()
        sheet = client.open(sheet_name)
        worksheet = sheet.worksheet(worksheet_name)

        # Print ALL sheet data before calling get_all_records()
        raw_values = worksheet.get_all_values()
        # print("✅ DEBUG: Raw Google Sheets Data:")
        # for row in raw_values:
            # print(row)

        # Now call get_all_records()
        records = worksheet.get_all_records()
        # print(f"✅ DEBUG: Retrieved records: {records}")

        if not records:
            logging.warning(f"⚠️ WARNING: No data found in '{sheet_name}' -> '{worksheet_name}'. Returning empty DataFrame.")
            return pd.DataFrame()

        return pd.DataFrame(records)

    except Exception as e:
        logging.error(f"❌ ERROR: Failed to read Google Sheets data - {e}")
        return pd.DataFrame()

# ✅ Write data to Google Sheets
def write_to_google_sheet(sheet_name, worksheet_name, dataframe):
    try:
        client = gspread.service_account()
        sheet = client.open(sheet_name)

        try:
            worksheet = sheet.worksheet(worksheet_name)
            worksheet.clear()  # ✅ Clear existing worksheet
        except gspread.exceptions.WorksheetNotFound:
            # logging.warning(f"⚠️ Worksheet '{worksheet_name}' not found in '{sheet_name}'. Creating a new one.")
            worksheet = sheet.add_worksheet(title=worksheet_name, rows="100", cols="20")

        # ✅ Ensure `append_rows()` is always executed
        worksheet.append_rows([dataframe.columns.tolist()] + dataframe.values.tolist())

    except Exception as e:
        logging.error(f"❌ ERROR: Failed to write to Google Sheets - {e}")
