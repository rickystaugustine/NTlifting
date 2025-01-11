import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Function to read data from Google Sheets
def read_google_sheets(sheet_name, worksheet_name):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name('/Users/ricky.staugustine/Documents/FB/Lifting/ntafterschoollifting-b8f7a5923646.json', scope)
    client = gspread.authorize(creds)

    sheet = client.open(sheet_name)
    worksheet = sheet.worksheet(worksheet_name)
    data = worksheet.get_all_records()
    return pd.DataFrame(data)

# Function to preprocess data
def preprocess_data(df, column_name):
    """
    Clean and preprocess the DataFrame by dropping rows with missing or invalid data.
    """
    if column_name in df.columns:
        original_len = len(df)
        df = df.dropna(subset=[column_name])  # Remove rows where the column is NaN
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')  # Convert to numeric
        df = df.dropna(subset=[column_name])  # Remove rows where conversion failed
        print(f"Cleaned '{column_name}': Removed {original_len - len(df)} invalid rows.")
    else:
        print(f"Column '{column_name}' not found in the DataFrame.")
    return df

# Alternative models
def logistic_model(reps, L, k, x0):
    return L / (1 + np.exp(-k * (reps - x0)))

def exponential_decay_model(reps, a, b, c):
    return a * np.exp(-b * reps) + c

def hybrid_model(reps, a, b, c, d):
    return a * np.log(reps + b) + c * reps + d

# Adjust the hybrid-log-exp model fitting with more refined initial guesses
def fit_models(program_df):
    # Extract Clean data
    clean_data = program_df[program_df['Exercise'] == 'Clean']
    reps = clean_data['# of Reps'].values
    percentages = clean_data['Multiplier of Max'].values

    results = {}

    # Cubic model
    def cubic_model(reps, a, b, c, d):
        return a * reps**3 + b * reps**2 + c * reps + d

    try:
        initial_guess = [
            np.ptp(percentages) / (np.ptp(reps)**3),  # Coefficient for reps^3
            -np.ptp(percentages) / (np.ptp(reps)**2), # Coefficient for reps^2
            np.mean(percentages) / np.ptp(reps),      # Coefficient for reps
            np.median(percentages)                   # Constant term
        ]
        popt, _ = curve_fit(cubic_model, reps, percentages, maxfev=10000, p0=initial_guess)
        results['cubic'] = (
            lambda reps, a=popt[0], b=popt[1], c=popt[2], d=popt[3]: cubic_model(reps, a, b, c, d),
            popt
        )
    except RuntimeError as e:
        print(f"Cubic Model failed to converge: {e}")

    # Hybrid-log-exp model
    def hybrid_log_exp_model(reps, a, b, c, d, e):
        return a * np.log(reps + b) + c * np.exp(-d * reps) + e

    try:
        initial_a = np.ptp(percentages) / np.log(np.ptp(reps) + 1)
        initial_b = max(0.1, -min(reps) + 0.1)
        initial_c = np.ptp(percentages) / np.exp(np.ptp(reps))
        initial_d = 0.1  # Decay rate for exponential term
        initial_e = np.mean(percentages)

        initial_guess = [initial_a, initial_b, initial_c, initial_d, initial_e]
        print(f"Initial guesses for hybrid-log-exp model: {initial_guess}")

        popt, _ = curve_fit(
            hybrid_log_exp_model,
            reps,
            percentages,
            maxfev=10000,
            p0=initial_guess,
            bounds=([-np.inf, 0.01, -np.inf, 0.01, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])
        )
        results['hybrid_log_exp'] = (
            lambda reps, a=popt[0], b=popt[1], c=popt[2], d=popt[3], e=popt[4]: hybrid_log_exp_model(reps, a, b, c, d, e),
            popt
        )
    except RuntimeError as e:
        print(f"Hybrid Log-Exp Model failed to converge: {e}")

    return results

# Cross-validation for model evaluation
def cross_validate_model(model_function, reps, percentages):
    reps_train, reps_val, percentages_train, percentages_val = train_test_split(reps, percentages, test_size=0.2, random_state=42)
    predictions = model_function(reps_val)
    mae = mean_absolute_error(percentages_val, predictions)
    return mae

# Identify outliers based on residuals
def identify_outliers(residuals, threshold=2.5):
    """
    Identify outliers in residuals based on a threshold of standard deviations.
    """
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    outliers = np.where(np.abs(residuals - mean_residual) > threshold * std_residual)[0]
    return outliers

# Simulate Clean Functional Max Calculation
def simulate_clean_functional_max(clean_function, program_df, maxes_df):
    simulated_results = []
    clean_data = program_df[program_df['Exercise'] == 'Clean']

    for _, program_row in clean_data.iterrows():
        assigned_reps = program_row['# of Reps']
        multiplier = program_row['Multiplier of Max']

        for _, maxes_row in maxes_df.iterrows():
            tested_max = maxes_row['Clean']

            if pd.isna(tested_max) or pd.isna(multiplier):
                print(f"Skipping row due to missing data: Tested Max={tested_max}, Multiplier={multiplier}")
                continue

            try:
                # Convert inputs to float
                tested_max = float(tested_max)
                multiplier = float(multiplier)

                assigned_weight = round(tested_max * multiplier / 5) * 5

                # Simulate actual data
                actual_reps = assigned_reps + np.random.choice([-1, 0, 1])
                weight_deviation = np.random.uniform(-0.05, 0.05)
                actual_weight = assigned_weight * (1 + weight_deviation)

                actual_percentage = clean_function(actual_reps)
                functional_max = actual_weight / actual_percentage

                simulated_results.append({
                    'Player': maxes_row['Player'],
                    'Exercise': 'Clean',
                    'Tested Max': tested_max,
                    'Actual Reps': actual_reps,
                    'Actual Weight': actual_weight,
                    'Functional Max': round(functional_max / 5) * 5
                })

            except ValueError as e:
                print(f"Skipping invalid data: Tested Max={tested_max}, Multiplier={multiplier}. Error: {e}")
                continue

    simulated_results = pd.DataFrame(simulated_results)

    # Evaluate model
    mae = mean_absolute_error(simulated_results['Tested Max'], simulated_results['Functional Max'])
    print(f"Mean Absolute Error (MAE) for Clean: {mae:.2f}")

    # Plot residuals
    residuals = simulated_results['Functional Max'] - simulated_results['Tested Max']
    outliers = identify_outliers(residuals)
    
    print(f"Identified Outliers: {len(outliers)} rows")
    
    plt.scatter(simulated_results['Tested Max'], residuals, label="Residuals")
    plt.axhline(0, color='r', linestyle='--', label="Zero Line")
    plt.scatter(simulated_results.iloc[outliers]['Tested Max'], residuals[outliers], color='orange', label="Outliers")
    plt.xlabel("Tested Max")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot for Clean")
    plt.legend()
    plt.show()

# Main function for Clean testing
def main():
    program_df = read_google_sheets('After-School Lifting', 'CompleteProgram')
    maxes_df = read_google_sheets('After-School Lifting', 'Maxes')

    # Preprocess the data to handle missing values
    maxes_df = preprocess_data(maxes_df, 'Clean')

    # Fit models for Clean
    model_results = fit_models(program_df)

    # Select the best model based on cross-validation
    best_model = None
    best_mae = float('inf')

    for model_name, (model_function, params) in model_results.items():
        clean_data = program_df[program_df['Exercise'] == 'Clean']
        reps = clean_data['# of Reps'].values
        percentages = clean_data['Multiplier of Max'].values
        mae = cross_validate_model(model_function, reps, percentages)
        print(f"{model_name.capitalize()} Model MAE: {mae:.4f}")

        if mae < best_mae:
            best_mae = mae
            best_model = model_function

    if best_model is not None:
        print(f"Best Model Selected with MAE: {best_mae:.4f}")
        # Simulate Functional Max for Clean
        simulate_clean_functional_max(best_model, program_df, maxes_df)

if __name__ == "__main__":
    main()
