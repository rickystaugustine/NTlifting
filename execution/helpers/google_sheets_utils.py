import gspread
import pandas as pd
import numpy as np
import logging
from oauth2client.service_account import ServiceAccountCredentials

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - INFO - %(message)s")

# Google Sheets API Authorization
def authorize_google_client():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name("/Users/ricky.staugustine/.config/NTlifting/ntafterschoollifting-b8f7a5923646.json", scope)
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
def write_to_google_sheet(sheet_name, worksheet_name, data):
    client = authorize_google_client()
    sheet = client.open(sheet_name)

    try:
        worksheet = sheet.worksheet(worksheet_name)
        worksheet.clear()
        logging.info(f"🔄 Existing worksheet {worksheet_name} cleared.")
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=worksheet_name, rows="1000", cols="26")
        logging.info(f"➕ Created new worksheet: {worksheet_name}")

    # Fix Pandas FutureWarnings
    data.infer_objects(copy=False)
    pd.set_option('future.no_silent_downcasting', True)
    data.fillna("NRM", inplace=True)  # Ensure missing max values remain "NRM"
    data = data.infer_objects(copy=False)  # Explicitly control type inference

    # Write data
    worksheet.update([data.columns.values.tolist()] + data.values.tolist())
    logging.info(f"✅ Successfully wrote {len(data)} rows to {sheet_name}/{worksheet_name}")
