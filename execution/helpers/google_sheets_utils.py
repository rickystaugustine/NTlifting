import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import os

def authorize_google_client():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        os.path.join("config", "ntafterschoollifting-b8f7a5923646.json"), scope
    )
    client = gspread.authorize(credentials)
    return client

def read_google_sheets(sheet_name: str, worksheet_name: str) -> pd.DataFrame:
    """
    Read data from a Google Sheet and return as a DataFrame.
    """
    client = authorize_google_client()
    sheet = client.open(sheet_name)
    worksheet = sheet.worksheet(worksheet_name)
    data = worksheet.get_all_records()
    return pd.DataFrame(data)

def write_to_google_sheet(sheet_name: str, worksheet_name: str, data: pd.DataFrame):
    """
    Write a DataFrame to a Google Sheet.
    """
    client = authorize_google_client()
    sheet = client.open(sheet_name)
    try:
        worksheet = sheet.worksheet(worksheet_name)
        worksheet.clear()
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=worksheet_name, rows="1000", cols="26")
    data = data.replace([np.inf, -np.inf, np.nan], 0)
    worksheet.update([data.columns.values.tolist()] + data.values.tolist())
