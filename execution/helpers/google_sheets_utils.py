import gspread
import pandas as pd
import numpy as np
import logging
import os
from oauth2client.service_account import ServiceAccountCredentials

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - INFO - %(message)s")

# Google Sheets API Authorization
def authorize_google_client():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    credentials_path = "/Users/ricky.staugustine/.config/NTlifting/ntafterschoollifting-b8f7a5923646.json"

    if not os.path.exists(credentials_path):
        logging.error(f"❌ ERROR: Google Sheets credentials file missing at {credentials_path}.")
        return None  # Prevent failure

    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    client = gspread.authorize(creds)
    logging.info("✅ Google Sheets API authorization successful!")
    return client

# Read data from Google Sheets
def read_google_sheets(sheet_name, worksheet_name):
    client = authorize_google_client()
    sheet = client.open(sheet_name)
    worksheet = sheet.worksheet(worksheet_name)
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)
    logging.info(f"✅ Successfully read {len(df)} rows from {sheet_name}/{worksheet_name}")
    return df

# Write data to Google Sheets with improved Pandas handling
import gspread
from gspread.exceptions import SpreadsheetNotFound

def write_to_google_sheet(worksheet_name, sheet_name, data):
    """Writes data to Google Sheets."""
    try:
        client = gspread.service_account()  # Ensure API authentication is set up
        sheet = client.open(sheet_name)
    except SpreadsheetNotFound:
        logging.error(f"❌ ERROR: Spreadsheet '{sheet_name}' not found. Creating a new one.")
        sheet = client.create(sheet_name)  # ✅ Create new sheet if missing

    worksheet = sheet.worksheet(worksheet_name)

    # Convert DataFrame to list if needed
    if hasattr(data, "values"):
        data = [data.columns.tolist()] + data.values.tolist()

    worksheet.clear()
    worksheet.append_rows(data)
    logging.info(f"✅ Successfully wrote data to {sheet_name}/{worksheet_name}.")
