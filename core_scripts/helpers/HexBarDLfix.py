import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
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

# Define the fitted curve function for Hex-Bar Deadlift
def hex_bar_deadlift_model(reps, a, b, c):
    return a * np.exp(-b * reps) + c

# Fitted parameters from Hex-Bar Deadlift model
hex_bar_params = [267.616918, -0.0000613491632, -266.950483]

# Test Hex-Bar Deadlift Model with Actual Data
def test_hex_bar_deadlift():
    # Load CompleteProgram data
    program_df = read_google_sheets('After-School Lifting', 'CompleteProgram')

    # Filter for Hex-Bar Deadlift data
    hex_bar_data = program_df[program_df['Exercise'] == 'Hex-Bar Deadlift']

    # Extract reps and multipliers from the dataset
    actual_reps = hex_bar_data['# of Reps'].values
    actual_multipliers = hex_bar_data['Multiplier of Max'].values

    # Predict multipliers using the fitted curve
    predicted_multipliers = hex_bar_deadlift_model(actual_reps, *hex_bar_params)

    # Calculate the Mean Absolute Error (MAE) for comparison
    mae = mean_absolute_error(actual_multipliers, predicted_multipliers)

    print(f"Mean Absolute Error (MAE) for Hex-Bar Deadlift on actual data: {mae:.2f}")

    # Plot actual vs predicted multipliers
    plt.scatter(actual_reps, actual_multipliers, label='Actual Data', color='blue')
    plt.plot(actual_reps, predicted_multipliers, label='Fitted Curve', color='red')
    plt.xlabel("Reps")
    plt.ylabel("Multiplier of Max")
    plt.title("Hex-Bar Deadlift: Actual vs Predicted Multipliers")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_hex_bar_deadlift()
