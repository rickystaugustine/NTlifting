import gspread
from google.oauth2.service_account import Credentials
import pandas as pd

# Constants
SERVICE_ACCOUNT_FILE = '/Users/ricky.staugustine/Documents/FB/ntafterschoollifting-b8f7a5923646.json'
CARDS_SPREADSHEET = 'After-School Lifting'
CORES_SPREADSHEET = 'Cores'

def authenticate_gspread():
    """Authenticate and return gspread client."""
    credentials = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return gspread.authorize(credentials)

def get_cores_data(gc):
    """Retrieve data from the Cores spreadsheet."""
    spreadsheet = gc.open(CORES_SPREADSHEET)
    exercise_tabs = ['Squat', 'Clean', 'Bench', 'Deadlift']
    cores_data = []

    for tab in exercise_tabs:
        worksheet = spreadsheet.worksheet(tab)
        data = worksheet.get_all_values()
        df = pd.DataFrame(data[1:], columns=data[0])  # Convert to DataFrame, skip headers
        df["Exercise"] = tab  # Add Exercise column
        cores_data.append(df)
    
    return pd.concat(cores_data, ignore_index=True)

def update_cards_weights(gc, cores_df):
    """Update the Weights column in the Cards tab."""
    cards_sheet = gc.open(CARDS_SPREADSHEET)
    cards_tab = cards_sheet.worksheet("Cards")
    cards_data = cards_tab.get_all_values()
    headers = cards_data[0]
    cards_df = pd.DataFrame(cards_data[1:], columns=headers)

    # Ensure numeric types for comparisons
    cores_df[["Week", "Set", "Reps"]] = cores_df[["Week", "Set", "Reps"]].apply(pd.to_numeric, errors="coerce")
    cards_df[["Week", "Set", "Reps"]] = cards_df[["Week", "Set", "Reps"]].apply(pd.to_numeric, errors="coerce")

    # Populate Weights column
    weights = []
    for _, row in cards_df.iterrows():
        exercise, week, set_num, reps = row["Exercise"], row["Week"], row["Set"], row["Reps"]
        weight = lookup_weight(cores_df, exercise, week, set_num, reps)
        weights.append(weight)
    
    cards_df["Weight"] = weights

    # Write updated data back to the Cards tab
    updated_data = [headers] + cards_df.values.tolist()
    cards_tab.clear()
    cards_tab.update(updated_data)
    print("Cards tab updated successfully.")

def lookup_weight(cores_df, exercise, week, set_num, reps):
    """Look up the weight from cores data."""
    match = cores_df[
        (cores_df["Exercise"] == exercise) &
        (cores_df["Week"] == week) &
        (cores_df["Set"] == set_num) &
        (cores_df["Reps"] == reps)
    ]
    return match["Weight [lb]"].iloc[0] if not match.empty else 0  # Return 0 if not found

def main():
    gc = authenticate_gspread()

    # Retrieve Cores data
    cores_df = get_cores_data(gc)

    # Update the Weights in the Cards tab
    update_cards_weights(gc, cores_df)

if __name__ == "__main__":
    main()

