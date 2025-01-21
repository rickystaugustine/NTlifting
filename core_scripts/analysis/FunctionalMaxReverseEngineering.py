import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, minimize, OptimizeWarning
from scipy.stats import binomtest
import warnings
import matplotlib.pyplot as plt

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

# Simulate Actual Lift Data with Normal Distributions
def simulate_actual_lift_data(assigned_weights_df):
    simulated_data = []

    # Group by exercise to calculate min/max values per exercise
    grouped = assigned_weights_df.groupby('Exercise')

    for exercise, group in grouped:
        min_r_assigned = group['# of Reps'].min()
        max_r_assigned = group['# of Reps'].max()
        min_w_assigned = group['Assigned Weight'].min()
        max_w_assigned = group['Assigned Weight'].max()

        # Calculate bounds for differences specific to the exercise
        min_r_diff = 0.1 * min_r_assigned - max_r_assigned
        max_r_diff = 3 * max_r_assigned - min_r_assigned
        min_w_diff = 0.5 * min_w_assigned - max_w_assigned
        max_w_diff = 1.5 * max_w_assigned - min_w_assigned

        for _, row in group.iterrows():
            player = row['Player']
            week = row['Week #']
            set_num = row['Set #']
            w_assigned = row['Assigned Weight']
            r_assigned = row['# of Reps']

            # Generate w_actual as a single normalized dataset
            w_diff = np.random.normal(0, 0.1 * w_assigned, size=1000)
            w_actual_candidates = w_assigned + w_diff
            w_actual_candidates = np.clip(w_actual_candidates, 0.5 * w_assigned, 1.5 * w_assigned)
            w_actual_candidates = (w_actual_candidates // 5) * 5  # Round to nearest 5 lbs
            w_actual = np.random.choice(w_actual_candidates)  # Randomly select one value

            # Generate r_actual with bounds dynamically calculated
            r_diff = np.random.normal(0, 0.5 * r_assigned, size=1000)
            r_actual_candidates = r_assigned + r_diff
            r_actual_candidates = np.clip(r_actual_candidates, 0.1 * r_assigned, 3 * r_assigned)

            # Separate into < and > r_assigned
            r_actual_low = r_actual_candidates[r_actual_candidates < r_assigned]
            r_actual_high = r_actual_candidates[r_actual_candidates >= r_assigned]

            # Balance the two distributions
            min_length = min(len(r_actual_low), len(r_actual_high))
            if min_length > 0:
                balanced_low = np.random.choice(r_actual_low, size=min_length, replace=False)
                balanced_high = np.random.choice(r_actual_high, size=min_length, replace=False)
                r_actual_combined = np.concatenate((balanced_low, balanced_high))
            else:
                r_actual_combined = r_actual_candidates  # Fallback to original candidates

            if len(r_actual_combined) == 0:
                r_actual = r_assigned  # Default to assigned if no valid values
            else:
                r_actual = np.random.choice(r_actual_combined)  # Randomly select one value

            r_actual = max(1, int(r_actual))  # Ensure integer reps

            simulated_data.append({
                'Player': player,
                'Exercise': exercise,
                'Week #': week,
                'Set #': set_num,
                'Assigned Weight': w_assigned,
                'Assigned Reps': r_assigned,
                'Actual Weight': w_actual,
                'Actual Reps': r_actual
            })

    return pd.DataFrame(simulated_data)

def analyze_simulated_data(simulated_data):
    # Analyze distributions of r_actual and w_actual

    # Descriptive statistics for r_actual
    r_diff = simulated_data['Actual Reps'] - simulated_data['Assigned Reps']
    print("r_actual - r_assigned stats:")
    print(r_diff.describe())

    # Proportion of r_actual < r_assigned vs r_actual >= r_assigned
    below_assigned = (simulated_data['Actual Reps'] < simulated_data['Assigned Reps']).sum()
    above_or_equal_assigned = (simulated_data['Actual Reps'] >= simulated_data['Assigned Reps']).sum()
    print(f"Proportion r_actual < r_assigned: {below_assigned / len(simulated_data):.2%}")
    print(f"Proportion r_actual >= r_assigned: {above_or_equal_assigned / len(simulated_data):.2%}")

    # Plot r_actual - r_assigned
    plt.figure()
    r_diff.hist(bins=30, alpha=0.7)
    plt.title("r_actual - r_assigned Distribution")
    plt.xlabel("r_actual - r_assigned")
    plt.ylabel("Frequency")
    plt.axvline(0, color='red', linestyle='dashed', linewidth=1, label='Assigned Value')
    plt.legend()
    plt.savefig("r_actual_minus_r_assigned_distribution.png")
    plt.close()

    # Descriptive statistics for W_actual
    w_diff = simulated_data['Actual Weight'] - simulated_data['Assigned Weight']
    print("W_actual - W_assigned stats:")
    print(w_diff.describe())

    # Plot W_actual - W_assigned
    plt.figure()
    w_diff.hist(bins=30, alpha=0.7)
    plt.title("W_actual - W_assigned Distribution")
    plt.xlabel("W_actual - W_assigned")
    plt.ylabel("Frequency")
    plt.axvline(0, color='red', linestyle='dashed', linewidth=1, label='Assigned Value')
    plt.legend()
    plt.savefig("W_actual_minus_w_assigned_distribution.png")
    plt.close()

    print("Simulated data analysis complete. See histograms for r_actual and W_actual differences.")

def calculate_assigned_weights(program_df, core_maxes_df, exercise_functions):
    assigned_weights = []

    for _, player_row in core_maxes_df.iterrows():
        player = player_row['Player']
        for _, program_row in program_df.iterrows():
            code = program_row['Code']
            week = program_row['Week #']
            set_num = program_row['Set #']
            reps = program_row['# of Reps']
            exercise = program_row['Exercise']
            relevant_core = program_row['Relevant Core']  # Use Relevant Core to find the max

            # Get the tested max for the relevant core
            tested_max = player_row.get(relevant_core, "0")
            try:
                tested_max = float(tested_max)
            except ValueError:
                tested_max = 0.0  # Default missing or invalid values to 0.0

            # Calculate the multiplier and assigned weight
            m_e = float(exercise_functions[code](week, set_num, reps))
            assigned_weight = m_e * tested_max

            assigned_weights.append({
                'Player': player,
                'Exercise': exercise,
                'Week #': week,
                'Set #': set_num,
                '# of Reps': reps,
                'Assigned Weight': (assigned_weight // 5) * 5
            })

    return pd.DataFrame(assigned_weights)

def main():
    program_df = read_google_sheets('After-School Lifting', 'CompleteProgram')
    core_maxes_df = read_google_sheets('After-School Lifting', 'Maxes')

    # Define exercise functions for M_e(w, s, r)
    def exercise_function(w, s, r):
        return 0.5 + 0.02 * w + 0.03 * s + 0.01 * r

    exercise_functions = {code: exercise_function for code in program_df['Code'].unique()}

    # Calculate Assigned Weights using Tested Maxes
    assigned_weights_df = calculate_assigned_weights(program_df, core_maxes_df, exercise_functions)
    write_to_google_sheet('After-School Lifting', 'AssignedWeights', assigned_weights_df)

    # Simulate Actual Lift Data
    simulated_data = simulate_actual_lift_data(assigned_weights_df)
    write_to_google_sheet('After-School Lifting', 'SimulatedLiftData', simulated_data)

    # Analyze Simulated Data
    analyze_simulated_data(simulated_data)

if __name__ == "__main__":
    main()
