import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from scipy.optimize import curve_fit


def read_google_sheets(sheet_name, worksheet_name):
    """
    Reads data from a Google Sheet.
    """
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        '/Users/ricky.staugustine/Documents/FB/Lifting/ntafterschoollifting-b8f7a5923646.json', scope)
    client = gspread.authorize(creds)

    sheet = client.open(sheet_name)
    worksheet = sheet.worksheet(worksheet_name)
    data = worksheet.get_all_records()
    return pd.DataFrame(data)


def simulate_actual_lifts(program_df, maxes_df, n_variants=1000):
    """
    Simulates ActualLifts data by introducing realistic deviations in weight and reps.
    """
    np.random.seed(42)
    simulated_data = []

    core_to_column = {
        'Bench': 'Bench',
        'Squat': 'Squat',
        'Clean': 'Clean',
        'Deadlift': 'Deadlift'
    }

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
                    assigned_weight = round(tested_max * multiplier / 5) * 5

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

    # Add Max Range column for grouping based on Tested Max values
    simulated_data['Max Range'] = pd.cut(
        simulated_data['Tested Max'],
        bins=[0, 200, 300, 500],
        labels=['Low', 'Medium', 'High']
    )

    return simulated_data

def fit_percentage_curve_by_exercise(program_df):
    """
    Fits a separate curve for each exercise based on reps and multipliers.
    """
    def rep_to_percentage(reps, a, b, c):
        return a * np.exp(-b * reps) + c

    exercise_functions = {}
    grouped_exercises = program_df.groupby('Exercise')

    # Create a single figure for diagnostics
    fig, axes = plt.subplots(len(grouped_exercises), 1, figsize=(8, len(grouped_exercises) * 4))

    if len(grouped_exercises) == 1:
        axes = [axes]

    for i, (exercise, group) in enumerate(grouped_exercises):
        grouped = group.groupby('# of Reps')['Multiplier of Max'].mean().reset_index()
        reps = grouped['# of Reps'].values
        percentages = grouped['Multiplier of Max'].values

        ax = axes[i]
        try:
            print(f"Fitting curve for {exercise}")
            print(grouped)
            if exercise in ["DB Reverse Lunge Step-up", "Chest-Supported DB Row", "1-Arm DB Row", "3-Way DB Shoulder Raise", "Bar RDL", "Hex-Bar Jump"]:
                coefficients = np.polyfit(reps, percentages, deg=1)
                exercise_functions[exercise] = np.poly1d(coefficients)
                ax.plot(reps, exercise_functions[exercise](reps), label="Linear Fit", color='green')
            else:
                weights = 1 / (percentages + 1e-6)
                popt, _ = curve_fit(rep_to_percentage, reps, percentages, sigma=weights, maxfev=10000, p0=[0.5, 0.1, 0.5])
                exercise_functions[exercise] = lambda reps, a=popt[0], b=popt[1], c=popt[2]: a * np.exp(-b * reps) + c
                ax.plot(reps, [exercise_functions[exercise](r) for r in reps], label="Exponential Fit", color='red')
        except RuntimeError as e:
            print(f"Curve fitting failed for {exercise}: {e}")
            coefficients = np.polyfit(reps, percentages, deg=2)
            exercise_functions[exercise] = np.poly1d(coefficients)

        ax.scatter(reps, percentages, label="Data")
        ax.set_title(f"Fitting Diagnostics for {exercise}")
        ax.legend()

    plt.tight_layout()
    plt.savefig("/Users/ricky.staugustine/Documents/Programming/GitHub/NTlifting/fitting_diagnostics.pdf")
    plt.close()

    return exercise_functions


def calculate_functional_max_with_simulation(simulated_data, percentage_functions, adjustment_coeff=0.002):
    """
    Calculates the functional max using exercise-specific percentage functions.
    """
    functional_max_results = []

    for _, row in simulated_data.iterrows():
        actual_weight = row['Actual Weight']
        actual_reps = row['Actual Reps']
        assigned_reps = row['Assigned Reps']
        relevant_core = row['Relevant Core']
        tested_max = row['Tested Max']
        exercise = row['Exercise']

        percentage_function = percentage_functions.get(exercise, lambda reps: 1)
        actual_percentage = percentage_function(actual_reps)

        rep_difference = actual_reps - assigned_reps
        adjustment_factor = 1 + (adjustment_coeff * np.sign(rep_difference) * np.sqrt(abs(rep_difference)))
        adjusted_percentage = actual_percentage / adjustment_factor

        functional_max = actual_weight / adjusted_percentage

        functional_max_results.append({
            'Player': row['Player'],
            'Relevant Core': relevant_core,
            'Tested Max': tested_max,
            'Calculated Functional Max': round(functional_max / 5) * 5
        })

    return pd.DataFrame(functional_max_results)


def evaluate_sensitivity(simulated_results):
    """
    Evaluates the accuracy and sensitivity of the calculated functional max.
    """
    tested_maxes = simulated_results['Tested Max']
    calculated_maxes = simulated_results['Calculated Functional Max']

    mae = mean_absolute_error(tested_maxes, calculated_maxes)
    r2 = r2_score(tested_maxes, calculated_maxes)

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R^2): {r2:.2f}")

    # Plot residuals
    residuals = calculated_maxes - tested_maxes
    plt.scatter(tested_maxes, residuals)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Tested Max")
    plt.ylabel("Residuals (Calculated Max - Tested Max)")
    plt.title("Residuals Plot")
    plt.savefig("Residuals.png")  # Save residuals plot
    plt.close()

    # Ensure 'Max Range' column exists
    if 'Max Range' not in simulated_results.columns:
        print("'Max Range' column is missing, re-adding it.")
        simulated_results['Max Range'] = pd.cut(
            simulated_results['Tested Max'], 
            bins=[0, 200, 300, 500], 
            labels=['Low', 'Medium', 'High']
        )

    # Analyze residuals by max range
    simulated_results['Residual'] = residuals
    residuals_by_range = simulated_results.groupby('Max Range')['Residual'].mean()
    print("Residuals by Max Range:")
    print(residuals_by_range)

    return mae, r2


def main():
    program_df = read_google_sheets('After-School Lifting', 'CompleteProgram')
    maxes_df = read_google_sheets('After-School Lifting', 'Maxes')

    simulated_actual_lifts = simulate_actual_lifts(program_df, maxes_df)

    percentage_functions = fit_percentage_curve_by_exercise(program_df)

    functional_max_results = calculate_functional_max_with_simulation(simulated_actual_lifts, percentage_functions)

    evaluate_sensitivity(functional_max_results)


if __name__ == "__main__":
    main()
