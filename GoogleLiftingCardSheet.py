import os
import pickle
import pandas as pd
import gspread
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Constants
SERVICE_ACCOUNT_FILE = '/Users/ricky.staugustine/Documents/FB/ntafterschoollifting-02762bc4807d.json'
CLIENT_SECRETS_FILE = '/Users/ricky.staugustine/Documents/FB/client_secret_743021829074-ta0ehdtfs0tnjgkaf998hu0otdki8mk8.apps.googleusercontent.com.json'
TOKEN_FILE = 'token.pickle'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# Global Data
players_max_cores = {}
e_to_core = {
    1: 'Squat', 2: 'Squat', 3: 'Squat',
    4: 'Clean', 5: 'Clean',
    6: 'Bench', 7: 'Bench', 8: 'Bench', 9: 'Bench', 10: 'Bench', 11: 'Bench',
    12: 'Deadlift', 13: 'Deadlift', 14: 'Deadlift'
}

# Authentication Functions
def authenticate_gspread(service_account_file):
    """Authenticate with Google Sheets API using a service account."""
    return gspread.service_account(filename=service_account_file)

def authenticate_google_sheets(creds_file, token_file=TOKEN_FILE, scopes=None):
    """Authenticate and return credentials for Google Sheets API."""
    if scopes is None:
        scopes = SCOPES
    
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            return pickle.load(token)
    
    flow = InstalledAppFlow.from_client_secrets_file(creds_file, scopes=scopes)
    creds = flow.run_local_server(port=0)

    with open(token_file, 'wb') as token:
        pickle.dump(creds, token)
    
    return creds

# Player Management
def add_player(name, bench, squat, clean, deadlift):
    """Add player maxes to the global dictionary."""
    players_max_cores[name] = {'Bench': bench, 'Squat': squat, 'Clean': clean, 'Deadlift': deadlift}

# Add players
add_player('Johnson', 250, 400, 175, 435)
add_player('Pearson', 155, 295, 125, 300)
add_player('Ott', 135, 250, 115, 245)
add_player('Marker', 135, 340, 160, 295)
add_player('Clark', 180, 300, 165, 330)
add_player('Wolfe', 85, 95, 85, 135)

# Percentage Calculation
def calculate_percentage(e, r, s, w):
    """Calculate the percentage for a given exercise."""
    formulas = {
        1: lambda: 0.0066482 * r + 0.0617036 * s + 0.03054017 * w + 0.5209833795013851,
        2: lambda: 0.003241 * r + 0.030518 * s + 0.01527008 * w + 0.26049168975069253,
        3: lambda: 0.075,
        4: lambda: -0.03675 * r + 0.03675 * w + 0.845833333333333,
        5: lambda: 0.01239669 * r + 0.1125 * s + 0.03533058 * w + 0.42004132231404967,
        6: lambda: 0.00698643 * r + 0.06477715 * s + 0.03207202 * w + 0.5469977839335181,
        7: lambda: 0.0066482 * r + 0.0617036 * s + 0.03054017 * w + 0.5209833795013851,
        8: lambda: 0.1375,
        9: lambda: 0.00265014 * r + 0.04158306 * s + 0.02095845 * w + 0.326088457987073,
        10: lambda: 0.125,
        11: lambda: 0.125,
        12: lambda: 0.25,
        13: lambda: 0.4,
        14: lambda: 0.0066482 * r + 0.0617036 * s + 0.03054017 * w + 0.5209833795013851
    }
    return formulas.get(e, lambda: None)()

# Weight Calculation
def calculate_weight(player, e, r, s, w):
    """Calculate the weight for a player's exercise."""
    core_lift = e_to_core.get(e)
    if not core_lift:
        return f"Invalid exercise code: {e}"
    
    player_max = players_max_cores.get(player, {}).get(core_lift)
    if player_max is None:
        return f"No max data for {player} and {core_lift}"
    
    percentage = calculate_percentage(e, r, s, w)
    if percentage is None:
        return "Invalid percentage calculation"
    
    weight = percentage * player_max
    return int((weight // 5) * 5)

# Data Processing
def load_exercise_data(file_path, columns):
    """Load exercise data from a CSV file."""
    return pd.read_csv(file_path, usecols=columns)

def process_exercise_data(df):
    """Convert DataFrame to a list of tuples."""
    return list(df.itertuples(index=False, name=None))

# Display Data
def display_exercise_weights(exercise_data):
    """
    Generate and display all exercise weights for players, grouped by player, week, exercise, and set.
    
    :param exercise_data: List of exercise tuples.
    :return: Grouped and sorted pandas DataFrame.
    """
    all_data = []
    for player in players_max_cores.keys():
        for exercise, e, r, s, w in exercise_data:
            weight = calculate_weight(player, e, r, s, w)
            all_data.append({
                "Player": player,
                "Exercise": exercise,
                "Week": w,
                "Set": s,
                "Reps": r,
                "Weight [lb]": weight
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Sort by Player, Week, Exercise, and Set
    df = df.sort_values(by=["Player", "Week", "Exercise", "Set"])
    
    return df

# Google Sheets Interaction
def update_sheet_with_data(gc, spreadsheet_name, sheet_name, data, headers):
    """
    Replace all existing data in a Google Sheet with new data.
    
    :param gc: Authenticated gspread client.
    :param spreadsheet_name: Name of the spreadsheet.
    :param sheet_name: Name of the sheet within the spreadsheet.
    :param data: Data to write (list of lists or DataFrame).
    :param headers: List of headers to add as the first row.
    """
    # Open spreadsheet and worksheet
    sh = gc.open(spreadsheet_name)
    worksheet = sh.worksheet(sheet_name)

    # Convert DataFrame to a list of lists if needed
    if isinstance(data, pd.DataFrame):
        data = data.values.tolist()

    # Clear the worksheet
    worksheet.clear()
    print("Sheet cleared successfully.")

    # Write headers and data
    worksheet.update([headers] + data)
    print("Data updated successfully.")

def apply_filter_to_table(service, spreadsheet_id, sheet_id, start_row=1, start_column=1, end_row=None, end_column=None):
    """
    Apply a filter to a range of data in a Google Sheet.
    
    :param service: Authenticated Google Sheets API service object.
    :param spreadsheet_id: The ID of the spreadsheet.
    :param sheet_id: The ID of the sheet where the filter is to be applied.
    :param start_row: The starting row index (1-based).
    :param start_column: The starting column index (1-based).
    :param end_row: The ending row index (1-based, inclusive), or None to include all rows.
    :param end_column: The ending column index (1-based, inclusive), or None to include all columns.
    """
    filter_request = {
        "requests": [
            {
                "setBasicFilter": {
                    "filter": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": start_row - 1,  # Convert to 0-based
                            "startColumnIndex": start_column - 1,  # Convert to 0-based
                            "endRowIndex": end_row if end_row else None,
                            "endColumnIndex": end_column if end_column else None
                        }
                    }
                }
            }
        ]
    }

    try:
        response = service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body=filter_request
        ).execute()
        print("Filter applied successfully.")
        return response
    except Exception as e:
        print(f"Failed to apply filter: {e}")
        return None

# Main Script
def main():
    # Authenticate with Google Sheets API
    creds = authenticate_google_sheets(CLIENT_SECRETS_FILE, TOKEN_FILE)
    service = build('sheets', 'v4', credentials=creds)
    gc = authenticate_gspread(SERVICE_ACCOUNT_FILE)

    # Spreadsheet details
    spreadsheet_id = '1EAv4dMF27XrPH9Mrs4X8Ahdk-HYjzPnPRcB_H1uJSBQ'  # Replace with the actual spreadsheet ID
    spreadsheet_name = 'After-School Lifting'  # Name of the spreadsheet
    sheet_name = 'Progrum'  # Name of the sheet within the spreadsheet

    # Load exercise data
    columns_to_read = ['Exercise', 'e', 'r', 's', 'w']
    df = load_exercise_data('/Users/ricky.staugustine/Documents/FB/After-School Lifting - Weights.csv', columns_to_read)
    exercise_data = process_exercise_data(df)

    # Display weights
    exercise_results_df = display_exercise_weights(exercise_data)

    # Define headers
    headers = ["Player", "Exercise", "Week", "Set", "Reps", "Weight [lb]"]

    # Replace the data in the sheet with the updated data
    update_sheet_with_data(gc, spreadsheet_name, sheet_name, exercise_results_df, headers=headers)

    # Retrieve the sheet ID
    spreadsheet_metadata = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheet_id = next(sheet['properties']['sheetId'] for sheet in spreadsheet_metadata['sheets']
                    if sheet['properties']['title'] == sheet_name)

    # Apply filter to the data range
    total_rows = len(exercise_results_df) + 1  # Add 1 for the header row
    total_columns = len(headers)
    apply_filter_to_table(
        service,
        spreadsheet_id=spreadsheet_id,
        sheet_id=sheet_id,
        start_row=1,
        start_column=1,
        end_row=total_rows,
        end_column=total_columns
    )

if __name__ == '__main__':
    main()
