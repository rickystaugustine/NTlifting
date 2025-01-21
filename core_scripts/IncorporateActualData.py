import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm
from joblib import Parallel, delayed
import warnings
import gspread  # For Google Sheets API
from oauth2client.service_account import ServiceAccountCredentials
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from datetime import datetime
from helpers.helpers import fit_single_exercise_global, simulate_iteration, generate_rep_differences_vectorized

# Setup
warnings.simplefilter("always", category=FutureWarning)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Google Sheets Integration
def authorize_google_client():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        '/Users/ricky.staugustine/Documents/FB/Lifting/ntafterschoollifting-b8f7a5923646.json', scope
    )
    client = gspread.authorize(credentials)
    return client

def read_google_sheets(sheet_name, worksheet_name):
    print(f"Reading data from Google Sheet: {sheet_name}, Worksheet: {worksheet_name}")
    client = authorize_google_client()
    sheet = client.open(sheet_name)
    worksheet = sheet.worksheet(worksheet_name)
    data = worksheet.get_all_records()
    print(f"Successfully read {len(data)} rows of data.")
    return pd.DataFrame(data)

def write_to_google_sheet(sheet_name, worksheet_name, data):
    print(f"Writing data to Google Sheet: {sheet_name}, Worksheet: {worksheet_name}")
    client = authorize_google_client()
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

# Preprocessing Functions
def preprocess_core_maxes(core_maxes_df):
    core_lifts = ["Bench", "Barbell Squat", "Clean", "Hex-Bar Deadlift"]
    flattened_core_maxes_df = core_maxes_df.melt(
        id_vars=["Player"],
        value_vars=core_lifts,
        var_name="Relevant Core",
        value_name="Tested Max"
    )
    # Ensure Tested Max is numeric
    flattened_core_maxes_df["Tested Max"] = pd.to_numeric(flattened_core_maxes_df["Tested Max"], errors="coerce").fillna(0)
    return flattened_core_maxes_df

def preprocess_experimental_data(experimental_data_df, program_df, core_maxes_df):
    # Map Relevant Core from program_df to experimental_data_df
    exercise_to_core = program_df.set_index("Exercise")["Relevant Core"].to_dict()
    experimental_data_df["Relevant Core"] = experimental_data_df["Exercise"].map(exercise_to_core)

    # Check for missing 'Relevant Core'
    if experimental_data_df['Relevant Core'].isnull().any():
        print("WARNING: Missing 'Relevant Core' in experimental_data_df")
        print(experimental_data_df[experimental_data_df['Relevant Core'].isnull()])

    # Merge Tested Max from core_maxes_df
    experimental_data_df = experimental_data_df.merge(
        core_maxes_df[["Player", "Relevant Core", "Tested Max"]],
        on=["Player", "Relevant Core"],
        how="left"
    )

    # Ensure Tested Max is numeric
    experimental_data_df["Tested Max"] = pd.to_numeric(
        experimental_data_df["Tested Max"], errors="coerce"
    ).fillna(0)

    # Debugging Output
    print("Preprocessed Experimental Data Columns:", experimental_data_df.columns)
    return experimental_data_df

# Core Functionalities
def fit_functions_by_exercise(program_df):
    exercises = program_df['Code'].unique()
    exercise_functions = {}
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(fit_single_exercise_global, int(code), program_df.to_dict("records")): int(code)
            for code in exercises
        }
        for future in as_completed(futures):
            code = futures[future]
            try:
                func = future.result()
                exercise_functions[code] = func
                print(f"Task completed for exercise code {code}.")
            except Exception as e:
                print(f"Error processing exercise code {code}: {e}")
    return exercise_functions

def create_repeated_program(program_df, core_maxes_df):
    """
    Create a repeated program DataFrame for all players.
    """
    # Get the list of players
    players = core_maxes_df["Player"].unique()

    # Repeat program_df for each player
    repeated_program_df = pd.concat(
        [program_df.assign(Player=player) for player in players],
        ignore_index=True
    )

    # Map Relevant Core dynamically
    exercise_to_core = program_df.set_index("Exercise")["Relevant Core"].to_dict()
    repeated_program_df["Relevant Core"] = repeated_program_df["Exercise"].map(exercise_to_core)

    return repeated_program_df

def calculate_assigned_weights_vectorized(repeated_program_df, flattened_core_maxes_df, exercise_functions):
    print("Merging tested max values...")

    repeated_program_df = repeated_program_df.merge(
        flattened_core_maxes_df,  # now has columns: Player, Relevant Core, Tested Max
        on=["Player", "Relevant Core"],
        how="left"
    )

    # Ensure 'Tested Max' is numeric
    repeated_program_df["Tested Max"] = pd.to_numeric(
        repeated_program_df["Tested Max"], errors="coerce"
    ).fillna(0)

    # Calculate multipliers
    weeks = repeated_program_df["Week #"].values
    sets = repeated_program_df["Set #"].values
    reps = repeated_program_df["# of Reps"].values
    codes = repeated_program_df["Code"].values

    multipliers = [
        exercise_functions[int(code)][1](w, s, r)
        for code, w, s, r in zip(codes, weeks, sets, reps)
    ]
    repeated_program_df["Multiplier"] = multipliers

    # Calculate Assigned Weight
    repeated_program_df["Assigned Weight"] = (
        np.floor(repeated_program_df["Tested Max"] * repeated_program_df["Multiplier"] / 5) * 5
    )

    return repeated_program_df

def simulate_actual_lift_data_optimized(assigned_weights_df, num_iterations=5):
    print("Preparing data for simulation...")

    # Repeat rows for each iteration
    num_rows = len(assigned_weights_df)
    repeated_indices = np.tile(np.arange(num_rows), num_iterations)
    repeated_data = assigned_weights_df.iloc[repeated_indices].reset_index(drop=True)

    # Ensure 'Relevant Core' is retained
    if 'Relevant Core' not in repeated_data.columns:
        raise KeyError("'Relevant Core' column is missing in repeated_data before simulation.")

    # Filter rows with valid weights
    repeated_data = repeated_data[repeated_data["Assigned Weight"] > 0]

    # Precompute necessary values
    r_assigned = repeated_data["# of Reps"].values
    w_assigned = repeated_data["Assigned Weight"].values

    print("Generating random samples for rep differences and weight adjustments...")

    # Vectorized computation for rep differences
    rep_differences = generate_rep_differences_vectorized(r_assigned)

    # Vectorized weight adjustments
    weight_differences = np.random.normal(loc=0, scale=0.1 * w_assigned, size=len(w_assigned))

    # Apply computed differences
    repeated_data["Actual Reps"] = r_assigned + rep_differences
    repeated_data["Actual Weight"] = w_assigned + weight_differences

    print("Simulation completed.")
    return repeated_data

def calculate_functional_maxes_rowlevel(simulated_data, bias_correction=0):
    print("Columns in simulated_data:", simulated_data.columns)
    if 'Relevant Core' not in simulated_data.columns:
        raise KeyError("'Relevant Core' column is missing in simulated_data.")
    
    # Check for null or missing values
    if simulated_data['Relevant Core'].isnull().any():
        print("WARNING: 'Relevant Core' contains null values in simulated_data.")
        print(simulated_data[simulated_data['Relevant Core'].isnull()])

    grouped = simulated_data.groupby(['Player', 'Relevant Core'])
    results = []
    for (player, core), group in grouped:
        try:
            tested_max = float(group['Tested Max'].iloc[0])  # Ensure Tested Max is a float
        except ValueError:
            print(f"[WARNING] Invalid Tested Max for Player: {player}, Core: {core}")
            tested_max = 0.0  # Fallback for invalid Tested Max

        for idx, row in group.iterrows():
            try:
                W_assigned = float(row['Assigned Weight'])
                W_actual = float(row['Actual Weight'])
                r_assigned = float(row['Assigned Reps'])
                r_actual = float(row['Actual Reps'])
                multiplier = float(row["Multiplier"])
                x_raw = W_actual / multiplier if r_actual == r_assigned else W_assigned / multiplier
                x_func = 0.5 * (tested_max + x_raw) + bias_correction  # Apply bias correction
                results.append({
                    "Player": player,
                    "Core": core,
                    "FunctionalMax_Row": x_func,
                    "Tested Max": tested_max
                })
            except ValueError as ve:
                print(f"[WARNING] Invalid data in row {idx} for Player: {player}, Core: {core}: {ve}")
    return pd.DataFrame(results)

def analyze_experimental_vs_simulated(rowlevel_df, experimental_data_df, corrected=False):
    experimental_results = rowlevel_df[rowlevel_df['Player'].isin(experimental_data_df['Player'])]
    experimental_results.loc[:, 'Residual'] = experimental_results['FunctionalMax_Row'] - experimental_results['Tested Max']
    plt.hist(experimental_results['Residual'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Residuals for Experimental Data" + (" (Corrected)" if corrected else ""))
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()

def calculate_bias(rowlevel_df, experimental_data_df):
    experimental_results = experimental_data_df.merge(
        rowlevel_df[['Player', 'Core', 'FunctionalMax_Row']],
        left_on=['Player', 'Relevant Core'],
        right_on=['Player', 'Core'],
        how='left'
    )

    experimental_results['Residual'] = experimental_results['FunctionalMax_Row'] - experimental_results['Tested Max']

    # Overall Bias
    overall_mean_bias = experimental_results['Residual'].mean()
    overall_median_bias = experimental_results['Residual'].median()

    # Bias by Core
    core_biases = experimental_results.groupby('Relevant Core')['Residual'].agg(['mean', 'median'])

    # Bias by Player
    player_biases = experimental_results.groupby('Player')['Residual'].agg(['mean', 'median'])

    print(f"Overall Mean Bias: {overall_mean_bias}, Median Bias: {overall_median_bias}")
    print(f"Bias by Core:\n{core_biases}")
    print(f"Bias by Player:\n{player_biases}")

    return overall_mean_bias, overall_median_bias, core_biases, player_biases

def weighted_mean(series, weights):
    return (series * weights).sum() / weights.sum()

def calculate_weighted_bias(rowlevel_df, experimental_data_df):
    experimental_results = experimental_data_df.merge(
        rowlevel_df[['Player', 'Core', 'FunctionalMax_Row']],
        left_on=['Player', 'Relevant Core'],
        right_on=['Player', 'Core'],
        how='left'
    )

    experimental_results['Residual'] = experimental_results['FunctionalMax_Row'] - experimental_results['Tested Max']

    # Calculate weights based on data count
    weights = experimental_results.groupby('Relevant Core')['Residual'].transform('count')

    # Weighted Mean and Median Bias
    weighted_mean_bias = weighted_mean(experimental_results['Residual'], weights)
    weighted_median_bias = experimental_results['Residual'].median()

    print(f"Weighted Mean Bias: {weighted_mean_bias}")
    print(f"Weighted Median Bias: {weighted_median_bias}")

    return weighted_mean_bias, weighted_median_bias

def calculate_time_based_bias(rowlevel_df, experimental_data_df):
    experimental_results = experimental_data_df.merge(
        rowlevel_df[['Player', 'Core', 'FunctionalMax_Row']],
        left_on=['Player', 'Relevant Core'],
        right_on=['Player', 'Core'],
        how='left'
    )

    experimental_results['Residual'] = experimental_results['FunctionalMax_Row'] - experimental_results['Tested Max']

    # Convert dates to datetime and group by time intervals (e.g., week)
    experimental_results['Date'] = pd.to_datetime(experimental_results['Date'])
    residuals_over_time = experimental_results.groupby(pd.Grouper(key='Date', freq='W'))['Residual'].mean()

    # Detect trends in residuals
    print("Residuals Over Time:")
    print(residuals_over_time)

    return residuals_over_time

def iterative_bias_correction(rowlevel_df, experimental_data_df, max_iterations=5, tolerance=0.1):
    for i in range(max_iterations):
        mean_bias, median_bias, core_biases, player_biases = calculate_bias(rowlevel_df, experimental_data_df)

        # Apply bias correction
        rowlevel_df = calculate_functional_maxes_rowlevel(
            rowlevel_df, bias_correction=median_bias
        )

        # Check if residuals are within tolerance
        residuals = rowlevel_df['FunctionalMax_Row'] - experimental_data_df['Tested Max']
        if residuals.abs().mean() < tolerance:
            print(f"Bias correction converged in {i+1} iterations.")
            break

    return rowlevel_df

# Main Function
def main():
    # Step 1: Read data from Google Sheets
    program_df = read_google_sheets("After-School Lifting", "CompleteProgram")
    core_maxes_df = read_google_sheets("After-School Lifting", "Maxes")
    experimental_data_df = read_google_sheets("After-School Lifting", "ActualData")

    # Step 2: Preprocess data
    core_maxes_df = preprocess_core_maxes(core_maxes_df)
    experimental_data_df = preprocess_experimental_data(experimental_data_df, program_df, core_maxes_df)
    print("Experimental Data Columns:", experimental_data_df.columns)
    repeated_program_df = create_repeated_program(program_df, core_maxes_df)

    # Step 3: Fit exercise functions
    exercise_functions = fit_functions_by_exercise(program_df)

    # Step 4: Calculate assigned weights
    assigned_weights_df = calculate_assigned_weights_vectorized(repeated_program_df, core_maxes_df, exercise_functions)

    print("Assigned weights columns:", assigned_weights_df.columns)

    # Step 5: Simulate actual lift data
    print("Columns before simulation:", assigned_weights_df.columns)
    simulated_data = simulate_actual_lift_data_optimized(assigned_weights_df)
    print("Columns after simulation:", simulated_data.columns)

    # Step 6: Combine simulated and experimental data
    # Combine simulated and experimental data
    combined_data = pd.concat([simulated_data, experimental_data_df], ignore_index=True)
    # Remove rows with missing 'Relevant Core'
    combined_data = combined_data.dropna(subset=['Relevant Core'])
    print("Combined Data Columns:", combined_data.columns)

    # Debugging: Validate 'Relevant Core' in combined_data
    if 'Relevant Core' not in combined_data.columns:
        raise KeyError("'Relevant Core' column is missing in combined_data.")
    if combined_data['Relevant Core'].isnull().any():
        print("WARNING: Missing 'Relevant Core' in combined_data rows.")
        print(combined_data[combined_data['Relevant Core'].isnull()])

    # Fill missing 'Relevant Core' if necessary
    combined_data["Relevant Core"] = combined_data["Relevant Core"].fillna("Unknown")

    if 'Relevant Core' not in combined_data.columns:
        raise KeyError("'Relevant Core' column is missing in combined_data.")

    if combined_data['Relevant Core'].isnull().any():
        print("WARNING: 'Relevant Core' contains null values.")
        print(combined_data[combined_data['Relevant Core'].isnull()])
        combined_data['Relevant Core'] = combined_data['Relevant Core'].fillna("Unknown")  # Default handling

    # Step 7: Calculate functional maxes from the combined data
    rowlevel_df = calculate_functional_maxes_rowlevel(combined_data)

    # Step 8: Calculate bias using experimental data
    print("Calculating bias...")
    mean_bias, median_bias, core_biases, player_biases = calculate_bias(rowlevel_df, experimental_data_df)

    # Step 9: Recalculate functional maxes with bias correction
    rowlevel_df = iterative_bias_correction(rowlevel_df, experimental_data_df, max_iterations=5, tolerance=0.1)

    # Step 10: Analyze and output results
    analyze_experimental_vs_simulated(rowlevel_df, experimental_data_df, corrected=True)
    write_to_google_sheet("After-School Lifting", "FunctionalMaxes", rowlevel_df)

    print("Process completed successfully.")

if __name__ == "__main__":
    main()
