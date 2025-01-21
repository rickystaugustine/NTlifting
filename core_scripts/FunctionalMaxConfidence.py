import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import norm
from scipy.integrate import quad
import warnings
import matplotlib.pyplot as plt
from FunctionalMaxEvaluation import fit_functions_by_exercise, calculate_assigned_weights, simulate_actual_lift_data
import time

def debug_timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"[DEBUG] Starting '{func.__name__}'...")
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"[DEBUG] Finished '{func.__name__}' in {elapsed:.2f} seconds.")
        return result
    return wrapper

# Google Sheets functions
@debug_timer
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

@debug_timer
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

@debug_timer
def calculate_confidence_intervals(df, group_col, value_col, confidence=0.95):
    """
    Calculate confidence intervals for a grouped column in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        group_col (str): Column to group by (e.g., 'Player').
        value_col (str): Column for which to compute the confidence intervals (e.g., 'FunctionalMax').
        confidence (float): Confidence level for the intervals (default is 0.95).
    
    Returns:
        pd.DataFrame: A DataFrame with lower and upper bounds for each group.
    """
    print("Calculating confidence intervals...")
    grouped = df.groupby(group_col)
    results = []

    for group_name, group_data in grouped:
        data = group_data[value_col]
        mean = np.mean(data)
        sem = np.std(data, ddof=1) / np.sqrt(len(data))  # Standard Error of the Mean
        margin = sem * norm.ppf((1 + confidence) / 2)
        lower = mean - margin
        upper = mean + margin
        results.append({
            group_col: group_name,
            'Mean': mean,
            'Lower CI': lower,
            'Upper CI': upper
        })

    return pd.DataFrame(results)

@debug_timer
def plot_residuals(df, actual_col, predicted_col, group_col):
    """
    Plot residuals to analyze the differences between actual and predicted values.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        actual_col (str): Column with actual values.
        predicted_col (str): Column with predicted values.
        group_col (str): Column to group residuals by (e.g., 'Player').
    """
    print("Plotting residuals...")
    df['Residual'] = df[actual_col] - df[predicted_col]
    plt.figure(figsize=(10, 6))
    for player, group_data in df.groupby(group_col):
        plt.scatter(group_data[predicted_col], group_data['Residual'], label=player, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title("Residuals Analysis")
    plt.xlabel("Predicted Functional Max")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.legend(title="Players", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("Residuals.png")
    plt.show()

@debug_timer
def calculate_weighted_error(df, error_col, weight_col):
    """
    Calculate a weighted error for the given error column.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        error_col (str): Column with error values (e.g., residuals).
        weight_col (str): Column with weights to apply.
    
    Returns:
        float: The weighted error.
    """
    print("Calculating weighted error...")
    total_weight = df[weight_col].sum()
    weighted_error = (df[error_col] * df[weight_col]).sum() / total_weight
    return weighted_error

def main():
    # Step 1: Read data
    start_time = time.time()
    program_df = read_google_sheets('After-School Lifting', 'CompleteProgram')
    core_maxes_df = read_google_sheets('After-School Lifting', 'Maxes')
    print(f"Reading CompleteProgram took {time.time() - start_time:.2f} seconds.")

    # Step 2: Fit functions and calculate weights
    start_time = time.time()
    exercise_functions = fit_functions_by_exercise(program_df)
    print(f"Fit Functions Execution Time: {time.time() - start_time:.2f} seconds.")
    assigned_weights_df = calculate_assigned_weights(program_df, core_maxes_df, exercise_functions)

    print(f"Program DF Rows: {len(program_df)}")
    print(f"Assigned Weights DF Rows: {len(assigned_weights_df)}")

    # Step 3: Simulate actual data
    start_time = time.time()
    simulated_data = simulate_actual_lift_data(assigned_weights_df, program_df, num_iterations=5)
    print(f"Simulated Data Rows: {len(simulated_data)}")
    print(f"Simulating lift data took {time.time() - start_time:.2f} seconds.")

    # Step 4: Residual analysis
    plot_residuals(simulated_data, actual_col='Actual Weight', predicted_col='Assigned Weight', group_col='Player')

    # Step 5: Confidence intervals
    confidence_intervals = calculate_confidence_intervals(simulated_data, group_col='Player', value_col='Actual Weight')
    write_to_google_sheet('After-School Lifting', 'ConfidenceIntervals', confidence_intervals)

    # Step 6: Weighted error aggregation
    weighted_error = calculate_weighted_error(simulated_data, error_col='Error', weight_col='Assigned Weight')
    print(f"Weighted Error: {weighted_error}")

    # Write results to Google Sheets
    write_to_google_sheet('After-School Lifting', 'AssignedWeights', assigned_weights_df)
    write_to_google_sheet('After-School Lifting', 'SimulatedLiftData', simulated_data)

if __name__ == "__main__":
    main()
