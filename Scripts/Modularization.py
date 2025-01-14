import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import norm
from scipy.integrate import quad
from joblib import Parallel, delayed
import multiprocessing
import warnings
import gspread  # For Google Sheets API
from oauth2client.service_account import ServiceAccountCredentials
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from concurrent.futures import ProcessPoolExecutor, as_completed
from helpers import fit_single_exercise_global, simulate_iteration, generate_rep_differences_vectorized  # Import the functions.

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

# Updated Function for Parallel Execution
def fit_functions_by_exercise(program_df):
    """
    Fit functions for all exercises using ProcessPoolExecutor.
    """
    exercises = program_df['Code'].unique()
    program_records = program_df.to_dict("records")  # Serialize program_df
    exercise_functions = {}

    print(f"Submitting {len(exercises)} tasks to ProcessPoolExecutor...")

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(fit_single_exercise_global, int(code), program_records): int(code)
            for code in exercises
        }

        for future in as_completed(futures):
            code = futures[future]
            try:
                func = future.result()  # Extract the callable function directly
                exercise_functions[int(code)] = func
                print(f"Task completed for exercise code {code}.")
            except Exception as e:
                print(f"Error processing exercise code {code}: {e}")

    if not exercise_functions:
        raise ValueError("No exercise functions could be fitted. Check the errors logged.")

    print("Exercise functions fitted successfully.")
    return exercise_functions
    
def preprocess_core_maxes(core_maxes_df):
    """
    Flatten the core_maxes_df for vectorized lookups.
    Returns a DataFrame instead of a dictionary for use with merge operations.
    """
    flattened_core_maxes = core_maxes_df.melt(
        id_vars=["Player"],
        var_name="Relevant Core",
        value_name="Relevant Core Max"  # Adjusted column name for clarity
    )
    return flattened_core_maxes

def create_repeated_program(program_df, core_maxes_df):
    """
    Create a repeated program DataFrame for all players.
    Flattening of core_maxes_df is now decoupled from this function.
    """
    # Get the list of players
    players = core_maxes_df["Player"].unique()

    # Repeat program_df for each player
    repeated_program_df = pd.concat(
        [program_df.assign(Player=player) for player in players],
        ignore_index=True
    )

    # Map Relevant Core dynamically (using a flattened dictionary elsewhere)
    exercise_to_core = program_df.set_index("Exercise")["Relevant Core"].to_dict()
    repeated_program_df["Relevant Core"] = repeated_program_df["Exercise"].map(exercise_to_core)

    return repeated_program_df

# Function: Calculate Assigned Weights
# Replace calculate_assigned_weights_vectorized to handle the new return format
def calculate_assigned_weights_vectorized(repeated_program_df, flattened_core_maxes_df, exercise_functions):
    """
    Vectorized calculation of assigned weights for all players iterating through the program.
    """
    print("Merging Relevant Core Max values...")

    # Merge `flattened_core_maxes_df` into `repeated_program_df`
    repeated_program_df = repeated_program_df.merge(
        flattened_core_maxes_df,
        on=["Player", "Relevant Core"],
        how="left",
        suffixes=("", "_from_core_maxes"),
    )

    # Consolidate `Relevant Core Max` columns
    if "Relevant Core Max_from_core_maxes" in repeated_program_df.columns:
        print("Consolidating duplicate `Relevant Core Max` columns...")
        repeated_program_df["Relevant Core Max"] = repeated_program_df.pop("Relevant Core Max_from_core_maxes")

    # Ensure `Relevant Core Max` exists
    if "Relevant Core Max" not in repeated_program_df.columns:
        print("WARNING: `Relevant Core Max` column is missing. Filling with zeros.")
        repeated_program_df["Relevant Core Max"] = 0.0

    # Ensure `Relevant Core Max` is numeric
    repeated_program_df["Relevant Core Max"] = pd.to_numeric(
        repeated_program_df["Relevant Core Max"], errors="coerce"
    ).fillna(0)

    print("Merged DataFrame (Sample Rows):")
    print(repeated_program_df[["Player", "Relevant Core", "Relevant Core Max"]].head())

    # Calculate multipliers for each row
    weeks = repeated_program_df["Week #"].values
    sets = repeated_program_df["Set #"].values
    reps = repeated_program_df["# of Reps"].values
    codes = repeated_program_df["Code"].values

    # Extract the callable object and call it
    multipliers = [
        exercise_functions[int(code)][1](w, s, r)  # Access the callable in the tuple
        for code, w, s, r in zip(codes, weeks, sets, reps)
    ]
    repeated_program_df["Multiplier"] = multipliers

    # Calculate assigned weights
    repeated_program_df["Assigned Weight"] = (
        np.floor(repeated_program_df["Relevant Core Max"] * repeated_program_df["Multiplier"] / 5) * 5
    )
    return repeated_program_df
    
# Adjust simulate_actual_lift_data_optimized
def simulate_actual_lift_data_optimized(assigned_weights_df, num_iterations=5):
    """
    Optimized simulation of actual lift data using vectorized computation.
    """
    print("Preparing data for simulation...")

    # Repeat rows for each iteration
    num_rows = len(assigned_weights_df)
    repeated_indices = np.tile(np.arange(num_rows), num_iterations)
    repeated_data = assigned_weights_df.iloc[repeated_indices].reset_index(drop=True)

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

def analyze_simulated_data(simulated_data, output_file="combined_plots.png"):
    grouped = simulated_data.groupby("Exercise")
    num_exercises = len(grouped)
    rows = int(np.ceil(np.sqrt(num_exercises)))
    cols = int(np.ceil(num_exercises / rows))

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=list(grouped.groups.keys()))

    row, col = 1, 1
    for exercise, group in grouped:
        rep_differences = group["Actual Reps"] - group["# of Reps"]
        fig.add_trace(go.Histogram(x=rep_differences, nbinsx=20, name=exercise), row=row, col=col)
        
        col += 1
        if col > cols:
            col = 1
            row += 1

    fig.update_layout(title="Rep Differences by Exercise", height=800, width=1000)
    fig.write_image(output_file)
    print(f"Combined plots saved to {output_file}.")

# Main Function
def main():
    # Step 1: Read input data
    print("Reading data from Google Sheets...")
    program_df = read_google_sheets("After-School Lifting", "CompleteProgram")
    core_maxes_df = read_google_sheets("After-School Lifting", "Maxes")

    # Step 2: Preprocess core_maxes_df
    print("Preprocessing core maxes...")
    flattened_core_maxes_df = preprocess_core_maxes(core_maxes_df)
    print(f"Preprocessed Core Maxes (Sample Rows):\n{flattened_core_maxes_df.head()}")

    # Step 3: Create repeated program
    print("Creating repeated program...")
    repeated_program_df = create_repeated_program(program_df, core_maxes_df)
    print(f"Repeated Program (Sample Rows):\n{repeated_program_df.head()}")

    # Step 4: Merge relevant core max values into the repeated program
    print("Mapping Relevant Core Max values...")
    repeated_program_df = repeated_program_df.merge(
        flattened_core_maxes_df,
        on=["Player", "Relevant Core"],
        how="left"
    )
    print(f"Relevant Core Max Mapping (Sample Rows):\n{repeated_program_df[['Player', 'Relevant Core', 'Relevant Core Max']].head()}")

    # Step 5: Fit exercise functions
    print("Fitting functions by exercise...")
    exercise_functions = fit_functions_by_exercise(program_df)

    # Debugging: Ensure exercise_functions are populated
    if not exercise_functions:
        print("WARNING: No exercise functions generated!")
    else:
        print(f"Exercise Functions Keys: {list(exercise_functions.keys())[:5]}")  # Log a sample of the keys

    # Step 6: Calculate multipliers and assigned weights
    print("Calculating multipliers and assigned weights...")
    repeated_program_df = calculate_assigned_weights_vectorized(
        repeated_program_df, flattened_core_maxes_df, exercise_functions
    )
    print(f"Assigned Weights (Sample Rows):\n{repeated_program_df[['Player', 'Exercise', 'Assigned Weight']].head()}")

    # Step 7: Simulate actual lift data
    print("Simulating actual lift data...")
    simulated_data = simulate_actual_lift_data_optimized(repeated_program_df, num_iterations=5)
    print(f"Simulated Data (Sample Rows):\n{simulated_data.head()}")

    # Step 8: Analyze simulated data
    print("Analyzing simulated data...")
    analyze_simulated_data(simulated_data)

    # Step 9: Write results to Google Sheets
    print("Writing results to Google Sheets...")
    write_to_google_sheet("After-School Lifting", "AssignedWeights", repeated_program_df)
    write_to_google_sheet("After-School Lifting", "SimulatedLiftData", simulated_data)

    print("Process completed successfully!")
if __name__ == "__main__":
    main()
