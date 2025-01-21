import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import norm
from scipy.integrate import quad
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

    def custom_pdf(x, r_assigned, side):
        """
        Custom PDF for the left and right tails.
        Tapers to 0 at bounds while peaking at r_assigned.
    
        Parameters:
            x (float): The value to evaluate the PDF.
            r_assigned (float): The assigned number of reps.
            side (str): 'left' or 'right' to indicate the tail.
    
        Returns:
            float: Probability density at x.
        """
        if side == "left":
            # Left PDF: Normal curve tapering to 0 at 0.1 * r_assigned
            sigma_L = 0.2 * r_assigned
            return norm.pdf(x, loc=r_assigned, scale=sigma_L) if x >= 0.1 * r_assigned else 0
        elif side == "right":
            # Right PDF: Normal curve tapering to 0 at 3 * r_assigned
            sigma_R = 0.3 * r_assigned
            return norm.pdf(x, loc=r_assigned, scale=sigma_R) if x <= 3 * r_assigned else 0
        return 0
    
    def normalize_pdf(r_assigned, size=1000):
        """
        Combine left and right PDFs, normalize, and sample values.
    
        Parameters:
            r_assigned (float): Assigned number of reps.
            size (int): Number of samples.
    
        Returns:
            np.ndarray: Sampled values from the custom distribution.
        """
        # Define bounds
        min_bound = 0.1 * r_assigned
        max_bound = 3.0 * r_assigned
    
        # Define the combined PDF
        def combined_pdf(x):
            if x < r_assigned:
                return custom_pdf(x, r_assigned, "left")
            else:
                return custom_pdf(x, r_assigned, "right")
    
        # Normalize the combined PDF
        normalization_constant, _ = quad(combined_pdf, min_bound, max_bound)
    
        def normalized_pdf(x):
            return combined_pdf(x) / normalization_constant
    
        # Generate samples using inverse transform sampling
        x_vals = np.linspace(min_bound, max_bound, size)
        pdf_vals = np.array([normalized_pdf(x) for x in x_vals])
        cdf_vals = np.cumsum(pdf_vals)
        cdf_vals /= cdf_vals[-1]  # Normalize CDF to range [0, 1]
    
        random_probs = np.random.uniform(0, 1, size)
        sampled_vals = np.interp(random_probs, cdf_vals, x_vals)
    
        return sampled_vals
     
    def weight_custom_pdf(w, w_assigned):
        """
        Custom PDF for Actual Weight.
        Peaks at W_assigned and tapers to 0 at bounds (0.5 * W_assigned, 1.5 * W_assigned).
    
        Parameters:
            w (float): The value to evaluate the PDF.
            w_assigned (float): The assigned weight.
    
        Returns:
            float: Probability density at w.
        """
        lower_bound = 0.5 * w_assigned
        upper_bound = 1.5 * w_assigned
        sigma = 0.1 * w_assigned  # Standard deviation
    
        if lower_bound <= w <= upper_bound:
            return np.exp(-((w - w_assigned) ** 2) / (2 * sigma ** 2))
        return 0

    def generate_actual_weights_custom(w_assigned, size=1):
        """
        Generate Actual Weights based on a custom PDF.
    
        Parameters:
            w_assigned (float): Assigned weight.
            size (int): Number of samples.
    
        Returns:
            np.ndarray: Sampled values.
        """
        lower_bound = 0.5 * w_assigned
        upper_bound = 1.5 * w_assigned
    
        # Normalize the custom PDF
        normalization_constant, _ = quad(lambda w: weight_custom_pdf(w, w_assigned), lower_bound, upper_bound)
    
        def normalized_pdf(w):
            return weight_custom_pdf(w, w_assigned) / normalization_constant
    
        # Generate samples using inverse transform sampling
        x_vals = np.linspace(lower_bound, upper_bound, 1000)
        pdf_vals = np.array([normalized_pdf(x) for x in x_vals])
        cdf_vals = np.cumsum(pdf_vals)
        cdf_vals /= cdf_vals[-1]  # Normalize CDF to range [0, 1]
    
        random_probs = np.random.uniform(0, 1, size)
        sampled_vals = np.interp(random_probs, cdf_vals, x_vals)
    
        return sampled_vals


    for _ in range(num_iterations):
        iteration_data = assigned_weights_df.copy()
        iteration_data['Actual Reps'] = iteration_data['# of Reps'].apply(lambda r: np.random.choice(normalize_pdf(r)))
        iteration_data['Actual Weight'] = iteration_data['Assigned Weight'].apply(lambda w: np.random.choice(normalize_pdf(w)))
        simulated_data = pd.concat([simulated_data, iteration_data], ignore_index=True)

    return simulated_data

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

    # Write results back to Google Sheets
    write_to_google_sheet('After-School Lifting', 'AssignedWeights', assigned_weights_df)
    write_to_google_sheet('After-School Lifting', 'SimulatedLiftData', simulated_data)

if __name__ == "__main__":
    main()
