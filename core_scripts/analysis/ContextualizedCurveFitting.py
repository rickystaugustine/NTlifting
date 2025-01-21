import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import warnings

# Google Sheets helper functions

def read_google_sheets(sheet_name, worksheet_name):
    # Set up the scope and credentials to access Google Sheets
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        '/Users/ricky.staugustine/Documents/FB/Lifting/ntafterschoollifting-b8f7a5923646.json', scope)
    client = gspread.authorize(creds)
    
    # Open the specified sheet and read the data from the given worksheet
    sheet = client.open(sheet_name)
    worksheet = sheet.worksheet(worksheet_name)
    data = worksheet.get_all_records()
    return pd.DataFrame(data)

def write_to_google_sheet(sheet_name, worksheet_name, data):
    # Set up the scope and credentials to access Google Sheets
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        '/Users/ricky.staugustine/Documents/FB/Lifting/ntafterschoollifting-b8f7a5923646.json', scope)
    client = gspread.authorize(creds)

    sheet = client.open(sheet_name)

    try:
        # Attempt to clear the existing worksheet
        worksheet = sheet.worksheet(worksheet_name)
        worksheet.clear()
    except gspread.exceptions.WorksheetNotFound:
        # If the worksheet does not exist, create a new one
        worksheet = sheet.add_worksheet(title=worksheet_name, rows="1000", cols="26")

    # Replace invalid data with zeros and update the worksheet
    data = data.replace([np.inf, -np.inf, np.nan], 0)
    worksheet.update([data.columns.values.tolist()] + data.values.tolist())

# Define dynamic multiplier functions
def fit_functions_by_exercise(program_df):
    exercises = program_df['Code'].unique()
    exercise_functions = {}

    with open('fitted_parameters.log', 'w') as log_file:
        for code in exercises:
            exercise_df = program_df[program_df['Code'] == code]

            weeks = exercise_df['Week #'].values
            sets = exercise_df['Set #'].values
            reps = exercise_df['# of Reps'].values
            multipliers = exercise_df['Multiplier of Max'].values

            def m_function(inputs, a, b, c, d):
                w, s, r = inputs
                return a * w + b * s + c * np.log(r + 1) + d

            inputs = np.vstack((weeks, sets, reps)).T

            initial_a = (multipliers.max() - multipliers.min()) / (weeks.max() - weeks.min() + 1e-5)
            initial_b = (multipliers.max() - multipliers.min()) / (sets.max() - sets.min() + 1e-5)
            initial_c = (multipliers.max() - multipliers.min()) / (np.log(reps.max() + 1) - np.log(reps.min() + 1) + 1e-5)
            initial_d = multipliers.mean()
            initial_guess = [initial_a, initial_b, initial_c, initial_d]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", OptimizeWarning)
                    popt, _ = curve_fit(m_function, (weeks, sets, reps), multipliers, p0=initial_guess, maxfev=5000)
                log_file.write(f"Fitted parameters for Code {code}: {popt}\n")
                exercise_functions[code] = lambda w, s, r, p=popt: m_function((w, s, r), *p)

                # Calculate and display accuracy for M_e
                predicted_m = np.array([m_function((w, s, r), *popt) for w, s, r in zip(weeks, sets, reps)])
                accuracy = 1 - np.mean(np.abs(predicted_m - multipliers) / multipliers)

                if 1 <= code <= 14:
                    terms = []
                    if popt[0] != 0:
                        terms.append(f"{popt[0]}*w")
                    if popt[1] != 0:
                        terms.append(f"{popt[1]}*s")
                    if popt[2] != 0:
                        terms.append(f"{popt[2]}*log(r+1)")
                    if popt[3] != 0:
                        terms.append(f"{popt[3]}")
                    equation = " + ".join(terms)
                    print(f"M_e(w, s, r) for Code {code}: {equation} (Accuracy: {accuracy:.2%})")
            except RuntimeError as err:
                log_file.write(f"Curve fitting failed for Code {code}: {err}. Using mean multiplier.\n")
                mean_multiplier = multipliers.mean()
                exercise_functions[code] = lambda w, s, r, m=mean_multiplier: m

    return exercise_functions

def calculate_assigned_weights(program_df, exercise_functions, core_maxes_df):
    core_map = {
        'Bench': 'Bench',
        'Squat': 'Squat',
        'Clean': 'Clean',
        'Deadlift': 'Deadlift',
    }

    assigned_weights = []

    for _, player_row in core_maxes_df.iterrows():
        player = player_row['Player']

        for _, program_row in program_df.iterrows():
            code, w, s, r, relevant_core = (
                program_row['Code'],
                program_row['Week #'],
                program_row['Set #'],
                program_row['# of Reps'],
                program_row['Relevant Core'],
            )

            core_column = core_map.get(relevant_core)
            if not core_column:
                raise ValueError(f"No mapping found for Relevant Core: {relevant_core}")

            core_max = player_row[core_column]

            try:
                core_max = float(core_max)
            except ValueError:
                core_max = 0

            if pd.isna(core_max) or core_max == 0:
                assigned_weight = 0
            else:
                m_e = exercise_functions[code](w, s, r)
                assigned_weight = m_e * core_max

            assigned_weights.append({
                'Player': player,
                'Code': code,
                'Week #': w,
                'Set #': s,
                '# of Reps': r,
                'Relevant Core': relevant_core,
                'Assigned Weight': (assigned_weight // 5) * 5,
            })

    return pd.DataFrame(assigned_weights)

def validate_functions(program_df, exercise_functions):
    mismatches = []

    for _, row in program_df.iterrows():
        code, w, s, r = row['Code'], row['Week #'], row['Set #'], row['# of Reps']
        actual_m = row['Multiplier of Max']
        calculated_m = exercise_functions[code](w, s, r)

        if abs(actual_m - calculated_m) > 0.05:
            mismatches.append((code, w, s, r, actual_m, calculated_m))

    if mismatches:
        print("Discrepancies found in M function calculations:")
        for mismatch in mismatches:
            print(f"Code: {mismatch[0]}, Week: {mismatch[1]}, Set: {mismatch[2]}, Reps: {mismatch[3]}\n"
                  f"Actual M: {mismatch[4]}, Calculated M: {mismatch[5]}\n")

# Main function

def main():
    program_df = read_google_sheets('After-School Lifting', 'CompleteProgram')
    core_maxes_df = read_google_sheets('After-School Lifting', 'Maxes')

    exercise_functions = fit_functions_by_exercise(program_df)

    validate_functions(program_df, exercise_functions)

    assigned_weights_df = calculate_assigned_weights(program_df, exercise_functions, core_maxes_df)

    write_to_google_sheet('After-School Lifting', 'AssignedWeights', assigned_weights_df)

if __name__ == "__main__":
    main()
