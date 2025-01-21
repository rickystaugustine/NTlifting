import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def read_google_sheets(sheet_name, worksheet_name):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name('/Users/ricky.staugustine/Documents/FB/Lifting/ntafterschoollifting-b8f7a5923646.json', scope)
    client = gspread.authorize(creds)
    
    sheet = client.open(sheet_name)
    worksheet = sheet.worksheet(worksheet_name)
    data = worksheet.get_all_records()
    return pd.DataFrame(data)

def write_to_google_sheet(sheet_name, worksheet_name, data):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name('/Users/ricky.staugustine/Documents/FB/Lifting/ntafterschoollifting-b8f7a5923646.json', scope)
    client = gspread.authorize(creds)

    sheet = client.open(sheet_name)

    try:
        worksheet = sheet.worksheet(worksheet_name)
        worksheet.clear()
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=worksheet_name, rows="1000", cols="26")

    worksheet.update([data.columns.values.tolist()] + data.values.tolist())

def merge_data(program_df, maxes_df):
    core_to_column = {
        'Bench': 'Bench',
        'Squat': 'Squat',
        'Clean': 'Clean',
        'Deadlift': 'Deadlift'
    }
    expanded_rows = []

    for _, exercise_row in program_df.iterrows():
        relevant_core = exercise_row['Relevant Core']
        multiplier = exercise_row['Multiplier of Max']

        if relevant_core in core_to_column:
            max_column = core_to_column[relevant_core]

            for _, player_row in maxes_df.iterrows():
                if pd.notna(player_row[max_column]):
                    calculated_weight = player_row[max_column] * multiplier
                    rounded_weight = round(calculated_weight / 5) * 5
                    expanded_row = exercise_row.copy()
                    expanded_row['Player'] = player_row['Player']
                    expanded_row['Max Lift'] = player_row[max_column]
                    expanded_row['Calculated Weight'] = rounded_weight
                    expanded_rows.append(expanded_row)

    return pd.DataFrame(expanded_rows)

def fit_percentage_curve(program_df):
    def rep_to_percentage(reps, a, b, c):
        return a * np.exp(-b * reps) + c

    grouped = program_df.groupby('# of Reps')['Multiplier of Max'].mean().reset_index()
    reps = grouped['# of Reps'].values
    percentages = grouped['Multiplier of Max'].values

    try:
        popt, _ = curve_fit(rep_to_percentage, reps, percentages, maxfev=2000)
        print("Fitted Equation Parameters: a={}, b={}, c={}".format(*popt))
        return lambda reps: rep_to_percentage(reps, *popt)
    except RuntimeError as e:
        print(f"Curve fitting failed: {e}")
        # Fall back to simpler model if fitting fails
        coefficients = np.polyfit(reps, percentages, deg=2)
        poly_function = np.poly1d(coefficients)
        print(f"Using fallback polynomial fit: coefficients={coefficients}")
        return lambda reps: poly_function(reps)

def calculate_functional_max(merged_data, actual_lifts_df, maxes_df, percentage_function):
    functional_maxes = maxes_df.copy()
    functional_maxes.set_index('Player', inplace=True)

    for _, row in actual_lifts_df.iterrows():
        exercise = row['Exercise']
        player = row['Player']
        actual_weight = row['Actual Weight']
        actual_reps = row['Actual Reps']

        merged_entry = merged_data[(merged_data['Exercise'] == exercise) & (merged_data['Player'] == player)]

        if not merged_entry.empty:
            assigned_percentage = merged_entry.iloc[0]['Multiplier of Max']
            relevant_core = merged_entry.iloc[0]['Relevant Core']

            print(f"Processing Actual Lift: Exercise={exercise}, Player={player}, Relevant Core={relevant_core}")
            print(f"Assigned Percentage: {assigned_percentage}")

            # Recalculate actual percentage using curve-fitting function
            actual_percentage = percentage_function(actual_reps)
            print(f"Actual Percentage (based on reps {actual_reps}): {actual_percentage}")

            functional_max = actual_weight / actual_percentage
            print(f"Calculated Functional Max: {functional_max}")

            functional_maxes.at[player, relevant_core] = round(functional_max / 5) * 5  # Round to nearest 5 lbs

    functional_maxes.reset_index(inplace=True)
    return functional_maxes

def display_exercise_weights(merged_data):
    if merged_data.empty:
        print("No data to display")
    else:
        print("Processed Exercise Weights:")
        print(merged_data.head())

def main():
    maxes_df = read_google_sheets('After-School Lifting', 'Maxes')
    print("Players Maxes DataFrame:")
    print(maxes_df.head())
    
    program_df = read_google_sheets('After-School Lifting', 'CompleteProgram')
    print("Complete Program DataFrame:")
    print(program_df.head())

    actual_lifts_df = read_google_sheets('After-School Lifting', 'ActualLifts')

    # Fit percentage curve
    percentage_function = fit_percentage_curve(program_df)

    # Calculate Functional Maxes
    merged_data = merge_data(program_df, maxes_df)
    functional_maxes_df = calculate_functional_max(merged_data, actual_lifts_df, maxes_df, percentage_function)

    # Update ProcessedWeights based on Functional Maxes
    updated_merged_data = merge_data(program_df, functional_maxes_df)
    display_exercise_weights(updated_merged_data)

    # Write to Google Sheets
    write_to_google_sheet('After-School Lifting', 'ProcessedWeights', updated_merged_data)
    write_to_google_sheet('After-School Lifting', 'FunctionalMaxes', functional_maxes_df)

if __name__ == "__main__":
    main()
