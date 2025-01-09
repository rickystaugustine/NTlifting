import os
import pickle
import pandas as pd
import gspread
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Constants
SERVICE_ACCOUNT_FILE = '/Users/ricky.staugustine/Documents/FB/ntafterschoollifting-b8f7a5923646.json'
CLIENT_SECRETS_FILE = '/Users/ricky.staugustine/Documents/FB/client_secret.json'
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
    """Generate and display all exercise weights for players."""
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
    return df.sort_values(by=["Player", "Week", "Exercise", "Set"])

def update_sheet_with_data(gc, spreadsheet_name, sheet_name, data, headers):
    """Update Google Sheets with data."""
    try:
        print(f"Opening spreadsheet: {spreadsheet_name}")
        sh = gc.open(spreadsheet_name)
        worksheet = sh.worksheet(sheet_name)
        print(f"Clearing worksheet: {sheet_name}")
        worksheet.clear()
        if isinstance(data, pd.DataFrame):
            data = data.values.tolist()
        print("Updating worksheet with data...")
        worksheet.update([headers] + data)
        print(f"Google Sheet '{sheet_name}' updated successfully.")
    except Exception as e:
        print(f"Error updating sheet: {e}")

def output_exercise_tabs(gc, spreadsheet_name, exercise_data):
    """Update or create a spreadsheet with tabs for each exercise and apply filter feature."""
    try:
        # Open or create the spreadsheet
        spreadsheet = None
        try:
            spreadsheet = gc.open(spreadsheet_name)
            print(f"Spreadsheet '{spreadsheet_name}' found. Updating existing sheet.")
        except gspread.SpreadsheetNotFound:
            print(f"Spreadsheet '{spreadsheet_name}' not found. Creating a new one.")
            spreadsheet = gc.create(spreadsheet_name)
            spreadsheet.share('richard.staugustine@gmail.com', perm_type='user', role='writer')

        # Create headers
        headers = ["Player", "Functional Max", "Week", "Set", "Reps", "Weight [lb]"]
        exercises = ['Squat', 'Clean', 'Bench', 'Deadlift']

        # Loop through each exercise
        for exercise in exercises:
            exercise_rows = []
            for player in players_max_cores.keys():
                for ex, e, r, s, w in exercise_data:
                    if e_to_core.get(e) == exercise:
                        weight = calculate_weight(player, e, r, s, w)
                        functional_max = players_max_cores[player][exercise]
                        exercise_rows.append([player, functional_max, w, s, r, weight])

            # Check if the worksheet exists
            worksheet = None
            try:
                worksheet = spreadsheet.worksheet(exercise)
                print(f"Sheet '{exercise}' exists. Clearing data...")
                worksheet.clear()
            except gspread.WorksheetNotFound:
                print(f"Sheet '{exercise}' does not exist. Creating a new one...")
                worksheet = spreadsheet.add_worksheet(title=exercise, rows="100", cols="20")

            # Update the worksheet with new data
            worksheet.update([headers] + exercise_rows)

            # Apply filter to the data range
            worksheet_id = worksheet._properties['sheetId']
            apply_filter_to_worksheet(SERVICE_ACCOUNT_FILE, spreadsheet.id, worksheet_id, len(exercise_rows) + 1, len(headers))

        print(f"Spreadsheet '{spreadsheet_name}' updated with filters applied to all tabs.")
    except Exception as e:
        print(f"Error updating exercise tabs: {e}")

from google.oauth2.service_account import Credentials

def apply_filter_to_worksheet(service_account_file, spreadsheet_id, worksheet_id, num_rows, num_columns):
    """Apply a filter to the worksheet's data range."""
    try:
        # Authenticate directly with Google Sheets API using service account credentials
        credentials = Credentials.from_service_account_file(
            service_account_file,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        service = build('sheets', 'v4', credentials=credentials)

        # Build the filter request
        filter_request = {
            "requests": [
                {
                    "setBasicFilter": {
                        "filter": {
                            "range": {
                                "sheetId": worksheet_id,
                                "startRowIndex": 0,  # Start from the first row
                                "startColumnIndex": 0,  # Start from the first column
                                "endRowIndex": num_rows,  # Include all rows
                                "endColumnIndex": num_columns  # Include all columns
                            }
                        }
                    }
                }
            ]
        }

        # Send the request to the Google Sheets API
        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body=filter_request
        ).execute()
        print(f"Filter applied to sheet ID {worksheet_id}.")
    except Exception as e:
        print(f"Error applying filter: {e}")

def main():
    creds = authenticate_google_sheets(CLIENT_SECRETS_FILE, TOKEN_FILE)
    gc = authenticate_gspread(SERVICE_ACCOUNT_FILE)
    columns_to_read = ['Exercise', 'e', 'r', 's', 'w']
    df = load_exercise_data('/Users/ricky.staugustine/Documents/FB/After-School Lifting - Weights.csv', columns_to_read)
    exercise_data = process_exercise_data(df)

    # Update the main sheet
    print("Updating main sheet...")
    spreadsheet_name = 'After-School Lifting'
    sheet_name = 'Progrum'
    exercise_results_df = display_exercise_weights(exercise_data)
    headers = ["Player", "Exercise", "Week", "Set", "Reps", "Weight [lb]"]
    update_sheet_with_data(gc, spreadsheet_name, sheet_name, exercise_results_df, headers)

    # Create new spreadsheet for exercises
    print("Creating new spreadsheet for exercises...")
    output_exercise_tabs(gc, "Cores", exercise_data)

if __name__ == '__main__':
    main()
