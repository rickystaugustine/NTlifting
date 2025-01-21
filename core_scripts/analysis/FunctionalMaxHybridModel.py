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
    core_to_column = {
        'Bench': 'Bench',
        'Squat': 'Squat',
        'Clean': 'Clean',
        'Deadlift': 'Deadlift'
    }

    simulated_data = []
    for _, program_row in program_df.iterrows():
        exercise = program_row['Exercise']
        relevant_core = program_row['Relevant Core']
        
        # Ensure multiplier is numeric
        try:
            multiplier = float(program_row['Multiplier of Max'])
        except ValueError:
            print(f"Invalid multiplier for exercise {exercise}. Skipping row.")
            continue

        for _, player_row in maxes_df.iterrows():
            # Fetch the tested max for the relevant core lift
            tested_max = player_row.get(core_to_column.get(relevant_core), None)
            
            # Ensure tested_max is numeric
            try:
                tested_max = float(tested_max)
            except (ValueError, TypeError):
                print(f"Invalid tested max for player {player_row['Player']}. Skipping row.")
                continue

            # Calculate assigned weight
            assigned_weight = round(tested_max * multiplier / 5) * 5
            
            # Simulate actual lifts
            for _ in range(n_variants):
                actual_weight = assigned_weight * (1 + np.random.uniform(-0.05, 0.05))
                simulated_data.append({
                    'Player': player_row['Player'],
                    'Exercise': exercise,
                    'Tested Max': tested_max,
                    'Calculated Functional Max': assigned_weight
                })

    return pd.DataFrame(simulated_data)

# Evaluate Model
def evaluate_model(results_df):
    """
    Calculates and displays MAE and R-squared values for the entire dataset,
    as well as the MAE for each exercise.
    """
    tested_maxes = results_df['Tested Max']
    calculated_maxes = results_df['Calculated Functional Max']

    # Overall evaluation
    mae = mean_absolute_error(tested_maxes, calculated_maxes)
    r2 = r2_score(tested_maxes, calculated_maxes)
    print(f"Overall Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Overall R-squared (R^2): {r2:.2f}")

    # Grouped MAE by exercise
    grouped_mae = (
        results_df.groupby('Exercise')
        .apply(lambda x: mean_absolute_error(x['Tested Max'], x['Calculated Functional Max']))
        .reset_index(name='MAE')
    )
    print("\nMAE by Exercise:")
    print(grouped_mae)

    # Optional: Save or display grouped MAE
    return grouped_mae

# Main function
def main():
    # Load data from Google Sheets
    program_df = read_google_sheets('After-School Lifting', 'CompleteProgram')
    maxes_df = read_google_sheets('After-School Lifting', 'Maxes')

    # Simulate data
    simulated_data = simulate_actual_lifts(program_df, maxes_df)

    # Evaluate model and output MAE per exercise
    mae_by_exercise = evaluate_model(simulated_data)

    # Optional: Save MAE results to a file
    mae_by_exercise.to_csv('mae_by_exercise.csv', index=False)

if __name__ == "__main__":
    main()
