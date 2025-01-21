import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import sys
import os

# Get the absolute path of the project root
PROJECT_ROOT = "/Users/ricky.staugustine/Documents/Programming/GitHub/NTlifting"

# Ensure Python can find core_scripts and helpers
sys.path.insert(0, os.path.join(PROJECT_ROOT, "core_scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "core_scripts/helpers"))

# Now import helpers
from helpers.my_helpers import fit_single_exercise_global, simulate_iteration, generate_rep_differences_vectorized

# Step 1: Read Google Sheets

def read_google_sheets(sheet_name, worksheet_name):
    """
    Reads data from a specified Google Sheets worksheet and returns it as a pandas DataFrame.
    """
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        '/Users/ricky.staugustine/Documents/FB/Lifting/ntafterschoollifting-b8f7a5923646.json', scope
    )
    client = gspread.authorize(creds)

    sheet = client.open(sheet_name)
    worksheet = sheet.worksheet(worksheet_name)
    data = worksheet.get_all_records()
    return pd.DataFrame(data)

# Step 2: Identify Optimal Equations

def find_optimal_equations(program_df):
    """
    Dynamically generates optimal equations for each exercise based on input data,
    ensuring predictions are accurate within ±0.05 of the actual multiplier.
    Iterates on model complexity until the results meet the requirements,
    focusing on rep progression while preprocessing week and set progression.
    """
    equations = {}
    mse_scores = {}
    tolerance = 0.05  # Tolerance for prediction accuracy

    # Preprocess data to sort by Week # and Set #
    program_df = program_df.sort_values(by=['Week #', 'Set #', '# of Reps'])

    for exercise, group in program_df.groupby('Exercise'):
        if group['Multiplier of Max'].nunique() == 1:
            # Static multiplier for the exercise
            multiplier = group['Multiplier of Max'].iloc[0]
            equations[exercise] = f"Static multiplier: {multiplier:.3f}"
            mse_scores[exercise] = 0.0
        else:
            # Start with a linear model
            X = group[['Week #', 'Set #', '# of Reps']]
            y = group['Multiplier of Max']

            for degree in range(1, 4):  # Try polynomial degrees 1 to 3
                model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
                model.fit(X, y)

                # Predict values and check tolerance
                predictions = model.predict(X)
                differences = np.abs(predictions - y)
                if np.all(differences <= tolerance):
                    # Verify rep progression
                    rep_order = np.argsort(X['# of Reps'])
                    rep_progression = np.all(np.diff(predictions[rep_order]) <= 0)

                    print(f"{exercise} - Rep Progression: {rep_progression}")

                    if rep_progression:
                        # Construct the equation
                        coefficients = model.named_steps['linearregression'].coef_
                        intercept = model.named_steps['linearregression'].intercept_
                        equations[exercise] = (
                            f"Polynomial Degree {degree} Model: Coefficients={coefficients}, Intercept={intercept:.3f}"
                        )
                        # Calculate mean squared error
                        mse_scores[exercise] = mean_squared_error(y, predictions)
                        break

            # Fallback to simpler linear model if polynomial models fail
            if exercise not in equations:
                print(f"{exercise}: Falling back to simpler linear model.")
                linear_model = LinearRegression()
                linear_model.fit(X, y)
                predictions = linear_model.predict(X)
                differences = np.abs(predictions - y)
                if np.all(differences <= tolerance):
                    coefficients = linear_model.coef_
                    intercept = linear_model.intercept_
                    equations[exercise] = (
                        f"Linear Model: Coefficients={coefficients}, Intercept={intercept:.3f}"
                    )
                    mse_scores[exercise] = mean_squared_error(y, predictions)

    return equations, mse_scores

# Step 3: Write Results to Google Sheets

def write_equations_to_google_sheet(sheet_name, worksheet_name, equations, mse_scores):
    """
    Writes the optimal equations and their MSE scores to a specified Google Sheets worksheet.
    """
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        '/Users/ricky.staugustine/Documents/FB/Lifting/ntafterschoollifting-b8f7a5923646.json', scope
    )
    client = gspread.authorize(creds)

    sheet = client.open(sheet_name)

    try:
        worksheet = sheet.worksheet(worksheet_name)
        worksheet.clear()
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=worksheet_name, rows="1000", cols="2")

    # Prepare data for writing
    data = [
        ["Exercise", "Optimal Equation", "MSE"]
    ]
    for exercise, equation in equations.items():
        data.append([exercise, equation, mse_scores[exercise]])

    # Write data
    worksheet.update(data)

# Step 4: Main Function

def main():
    """
    Main execution function for identifying optimal equations and writing results.
    """
    program_df = read_google_sheets('After-School Lifting', 'CompleteProgram')
    print("Complete Program DataFrame:")
    print(program_df.head())

    # Find optimal equations
    equations, mse_scores = find_optimal_equations(program_df)

    # Print equations and MSE scores for verification
    for exercise, equation in equations.items():
        print(f"{exercise}: {equation} (MSE: {mse_scores[exercise]:.5f})")

    # Write results to Google Sheets
    write_equations_to_google_sheet('After-School Lifting', 'OptimalEquations', equations, mse_scores)

if __name__ == "__main__":
    main()
