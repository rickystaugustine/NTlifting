import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from scipy.optimize import curve_fit

# Function to read data from Google Sheets
def read_google_sheets(sheet_name, worksheet_name):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name('/Users/ricky.staugustine/Documents/FB/Lifting/ntafterschoollifting-b8f7a5923646.json', scope)
    client = gspread.authorize(creds)

    sheet = client.open(sheet_name)
    worksheet = sheet.worksheet(worksheet_name)
    data = worksheet.get_all_records()
    return pd.DataFrame(data)

# Simulate Actual Lifts Data
def simulate_actual_lifts(program_df, maxes_df, n_variants=1000):
    """
    Simulates ActualLifts data by introducing realistic deviations in weight and reps.
    """
    # Ensure Tested Max columns are numeric
    core_to_column = {
        'Bench': 'Bench',
        'Squat': 'Squat',
        'Clean': 'Clean',
        'Deadlift': 'Deadlift'
    }
    for core in core_to_column.values():
        if core in maxes_df.columns:
            maxes_df[core] = pd.to_numeric(maxes_df[core], errors='coerce')

    simulated_data = []
    for _, program_row in program_df.iterrows():
        exercise = program_row['Exercise']
        relevant_core = program_row['Relevant Core']
        assigned_reps = program_row['# of Reps']
        multiplier = program_row['Multiplier of Max']

        if relevant_core in core_to_column:
            max_column = core_to_column[relevant_core]

            for _, player_row in maxes_df.iterrows():
                tested_max = player_row[max_column]
                if pd.notna(tested_max):
                    try:
                        assigned_weight = round(tested_max * multiplier / 5) * 5
                    except TypeError as e:
                        print(f"Error in calculating assigned weight for player: {player_row['Player']}, exercise: {exercise}. {e}")
                        continue

                    for _ in range(n_variants):
                        actual_reps = assigned_reps + np.random.choice([-1, 0, 1])
                        weight_deviation = np.random.uniform(-0.05, 0.05)  # Reduced deviation for better precision
                        actual_weight = assigned_weight * (1 + weight_deviation)

                        simulated_data.append({
                            'Player': player_row['Player'],
                            'Exercise': exercise,
                            'Relevant Core': relevant_core,
                            'Assigned Reps': assigned_reps,
                            'Assigned Weight': assigned_weight,
                            'Actual Reps': actual_reps,
                            'Actual Weight': actual_weight,
                            'Tested Max': tested_max
                        })

    simulated_data = pd.DataFrame(simulated_data)
    # Add Max Range column for grouping
    simulated_data['Max Range'] = pd.cut(simulated_data['Tested Max'], bins=[0, 200, 300, 500], labels=['Low', 'Medium', 'High'])
    return simulated_data

# Fit percentage-to-reps curves
def fit_percentage_curve(program_df):
    def rep_to_percentage(reps, a, b, c):
        return a * np.exp(-b * reps) + c

    exercise_functions = {}
    for exercise, group in program_df.groupby('Exercise'):
        reps = group['# of Reps'].values
        percentages = group['Multiplier of Max'].values

        try:
            weights = 1 / (percentages + 1e-6)
            popt, _ = curve_fit(rep_to_percentage, reps, percentages, sigma=weights, maxfev=10000, p0=[0.5, 0.1, 0.5])
            exercise_functions[exercise] = lambda reps, a=popt[0], b=popt[1], c=popt[2]: a * np.exp(-b * reps) + c
        except RuntimeError:
            coefficients = np.polyfit(reps, percentages, deg=2)
            exercise_functions[exercise] = np.poly1d(coefficients)

    return exercise_functions

# Calculate Functional Max
def calculate_functional_max(simulated_data, percentage_functions):
    functional_max_results = []

    for _, row in simulated_data.iterrows():
        actual_weight = row['Actual Weight']
        actual_reps = row['Actual Reps']
        exercise = row['Exercise']

        percentage_function = percentage_functions.get(exercise, lambda reps: 1)
        actual_percentage = percentage_function(actual_reps)

        functional_max = actual_weight / actual_percentage

        functional_max_results.append({
            'Player': row['Player'],
            'Exercise': exercise,
            'Tested Max': row['Tested Max'],
            'Calculated Functional Max': round(functional_max / 5) * 5
        })

    return pd.DataFrame(functional_max_results)

# Evaluate Model
def evaluate_model(results_df):
    tested_maxes = results_df['Tested Max']
    calculated_maxes = results_df['Calculated Functional Max']

    mae = mean_absolute_error(tested_maxes, calculated_maxes)
    r2 = r2_score(tested_maxes, calculated_maxes)

    print(f"Overall Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Overall R-squared (R^2): {r2:.2f}")

    # MAE by Exercise
    mae_by_exercise = results_df.groupby('Exercise').apply(lambda group: mean_absolute_error(group['Tested Max'], group['Calculated Functional Max']))
    print("\nMAE by Exercise:")
    print(mae_by_exercise)

    residuals = calculated_maxes - tested_maxes
    plt.scatter(tested_maxes, residuals)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Tested Max")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.show()

# Main function
def main():
    program_df = read_google_sheets('After-School Lifting', 'CompleteProgram')
    maxes_df = read_google_sheets('After-School Lifting', 'Maxes')

    simulated_data = simulate_actual_lifts(program_df, maxes_df)
    percentage_functions = fit_percentage_curve(program_df)
    functional_max_results = calculate_functional_max(simulated_data, percentage_functions)

    evaluate_model(functional_max_results)

if __name__ == "__main__":
    main()
