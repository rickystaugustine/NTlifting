import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
import matplotlib.pyplot as plt

def read_google_sheets(sheet_name, worksheet_name):
    print(f"Reading data from Google Sheet: {sheet_name}, Worksheet: {worksheet_name}")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        '/Users/ricky.staugustine/Documents/FB/Lifting/ntafterschoollifting-b8f7a5923646.json', scope)
    client = gspread.authorize(creds)

    sheet = client.open(sheet_name)
    worksheet = sheet.worksheet(worksheet_name)
    data = worksheet.get_all_records()
    print(f"Successfully read {len(data)} rows of data.")
    return pd.DataFrame(data)

def write_to_google_sheet(sheet_name, worksheet_name, data):
    print(f"Writing data to Google Sheet: {sheet_name}, Worksheet: {worksheet_name}")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        '/Users/ricky.staugustine/Documents/FB/Lifting/ntafterschoollifting-b8f7a5923646.json', scope)
    client = gspread.authorize(creds)

    sheet = client.open(sheet_name)

    try:
        worksheet = sheet.worksheet(worksheet_name)
        worksheet.clear()
        print(f"Cleared existing worksheet: {worksheet_name}")
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=worksheet_name, rows="1000", cols="26")
        print(f"Created new worksheet: {worksheet_name}")

    data = data.replace([np.inf, -np.inf, np.nan], 0)
    worksheet.update([data.columns.values.tolist()] + data.values.tolist())
    print(f"Successfully wrote {len(data)} rows of data.")

def fit_functions_by_exercise(program_df):
    exercises = program_df['Code'].unique()
    exercise_functions = {}

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
        initial_guess = [1, 1, 1, multipliers.mean()]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, _ = curve_fit(m_function, (weeks, sets, reps), multipliers, p0=initial_guess, maxfev=5000)
            exercise_functions[code] = lambda w, s, r, p=popt: m_function((w, s, r), *p)
        except RuntimeError:
            exercise_functions[code] = lambda w, s, r, m=multipliers.mean(): m

    return exercise_functions

def calculate_assigned_weights(program_df, core_maxes_df, exercise_functions):
    assigned_weights = []

    for _, player_row in core_maxes_df.iterrows():
        player_name = player_row['Player']
        for _, program_row in program_df.iterrows():
            # Fetch the relevant core max and handle missing values
            core_max = pd.to_numeric(player_row.get(program_row['Relevant Core'], np.nan), errors='coerce')
            if np.isnan(core_max):
                # Skip this entry if no valid core max is available
                continue

            # Calculate the assigned weight
            assigned_weight = core_max * exercise_functions[program_row['Code']](
                program_row['Week #'], program_row['Set #'], program_row['# of Reps']
            )

            # Round down to the nearest 5 lbs
            assigned_weight = np.floor(assigned_weight / 5) * 5

            assigned_weights.append({
                'Player': player_name,
                'Exercise': program_row['Exercise'],  # Replace Code with Exercise
                **program_row.to_dict(),
                'Assigned Weight': assigned_weight,
                'Relevant Core Max': core_max  # Add Relevant Core Max column
            })

    return pd.DataFrame(assigned_weights)

def simulate_actual_lift_data(assigned_weights_df, program_df, num_iterations=5):
    np.random.seed(42)  # For reproducibility
    simulated_data = pd.DataFrame()

    def generate_actual_reps(row):
        r_assigned = row['# of Reps']
        if np.random.rand() < 0.5:  # Lower range
            r_actual_low = np.random.uniform(0.1 * r_assigned, 0.9 * r_assigned)
            return r_actual_low
        else:  # Higher range
            r_actual_high = np.random.uniform(1.0 * r_assigned, 3.0 * r_assigned)
            return r_actual_high

    def generate_actual_weights(row):
        weight_bounds = {
            'min': 0.5 * row['Assigned Weight'],
            'max': 1.5 * row['Assigned Weight']
        }
        return np.random.uniform(weight_bounds['min'], weight_bounds['max'])

    for _ in range(num_iterations):
        iteration_data = assigned_weights_df.copy()
        iteration_data['Actual Reps'] = iteration_data.apply(generate_actual_reps, axis=1)
        iteration_data['Actual Weight'] = iteration_data.apply(generate_actual_weights, axis=1)
        simulated_data = pd.concat([simulated_data, iteration_data], ignore_index=True)

    return simulated_data

def analyze_simulated_data(simulated_data):
    grouped_exercises = simulated_data.groupby('Exercise')
    num_exercises = len(grouped_exercises)

    rows = int(np.ceil(np.sqrt(num_exercises)))
    cols = int(np.ceil(num_exercises / rows))

    # Plot Rep Differences
    fig_reps, axs_reps = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    fig_reps.suptitle("Rep Difference Distribution", fontsize=16)
    axs_reps = axs_reps.flatten()

    for i, (exercise, group) in enumerate(grouped_exercises):
        rep_diff = group['Actual Reps'] - group['# of Reps']

        axs_reps[i].hist(rep_diff, bins=20, alpha=0.7, label=f'Rep Diff ({exercise})')
        axs_reps[i].set_title(f"{exercise}")
        axs_reps[i].set_xlabel("Difference in Reps")
        axs_reps[i].set_ylabel("Frequency")
        axs_reps[i].legend()

    # Hide unused subplots
    for j in range(i + 1, len(axs_reps)):
        axs_reps[j].axis('off')

    # Save the figure
    fig_reps.tight_layout(rect=[0, 0, 1, 0.95])
    fig_reps.savefig("Rep_Differences.png")
    plt.close(fig_reps)

def map_relevant_core(simulated_data, program_df):
    exercise_to_core = program_df.set_index('Exercise')['Relevant Core'].to_dict()
    simulated_data['Relevant Core'] = simulated_data['Exercise'].map(exercise_to_core)
    return simulated_data

def main():
   # Read data from Google Sheets
    program_df = read_google_sheets('After-School Lifting', 'CompleteProgram')
    core_maxes_df = read_google_sheets('After-School Lifting', 'Maxes')

    exercise_functions = fit_functions_by_exercise(program_df)

    # Calculate assigned weights
    assigned_weights_df = calculate_assigned_weights(program_df, core_maxes_df, exercise_functions)

    # Simulate actual lift data
    simulated_data = simulate_actual_lift_data(assigned_weights_df, program_df, num_iterations=5)

    # Analyze simulated data
    analyze_simulated_data(simulated_data)

    # Write results back to Google Sheets
    write_to_google_sheet('After-School Lifting', 'AssignedWeights', assigned_weights_df)
    write_to_google_sheet('After-School Lifting', 'SimulatedLiftData', simulated_data)

if __name__ == "__main__":
    main()
