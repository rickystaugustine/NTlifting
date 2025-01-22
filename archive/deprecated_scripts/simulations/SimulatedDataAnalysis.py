import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import norm
from scipy.integrate import quad

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

    def fit_single_exercise(code):
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
            return code, lambda w, s, r, p=popt: m_function((w, s, r), *p)
        except RuntimeError:
            return code, lambda w, s, r, m=multipliers.mean(): m

    results = Parallel(n_jobs=-1)(delayed(fit_single_exercise)(code) for code in exercises)
    exercise_functions.update(dict(results))

    return exercise_functions

def calculate_assigned_weights(program_df, core_maxes_df, exercise_functions):
    merged_df = core_maxes_df.merge(program_df, how='cross')

    def calculate_row(row):
        core_max = pd.to_numeric(row[row['Relevant Core']], errors='coerce')
        if np.isnan(core_max):
            return None

        assigned_weight = core_max * exercise_functions[row['Code']](
            row['Week #'], row['Set #'], row['# of Reps']
        )
        assigned_weight = np.floor(assigned_weight / 5) * 5
        row['Assigned Weight'] = assigned_weight
        row['Relevant Core Max'] = core_max
        return row

    calculated_weights = merged_df.apply(calculate_row, axis=1)
    return calculated_weights.dropna().reset_index(drop=True)

def generate_rep_differences_vectorized(r_assigned_array):
    """
    Generate rep differences for an array of assigned reps using a custom PDF.

    Parameters:
        r_assigned_array (np.ndarray): Array of assigned reps.

    Returns:
        np.ndarray: Array of rep differences.
    """
    size = len(r_assigned_array)
    sigma_L = 0.2991 * r_assigned_array
    sigma_R = 0.997 * r_assigned_array

    min_bound = -0.9 * r_assigned_array
    max_bound = 3.0 * r_assigned_array

    def combined_pdf(x, r, sigma_L, sigma_R):
        """
        Combined PDF for left and right tails, normalized.

        Parameters:
            x (float): Value to evaluate the PDF.
            r (float): Assigned reps.
            sigma_L (float): Std deviation for the left side.
            sigma_R (float): Std deviation for the right side.
        """
        if x < 0:
            return norm.pdf(x, loc=0, scale=sigma_L) if x >= -0.9 * r else 0
        else:
            return norm.pdf(x, loc=0, scale=sigma_R) if x <= 3 * r else 0

    def normalize_pdf(r, sigma_L, sigma_R):
        """
        Normalize the PDF for a single r_assigned value.

        Parameters:
            r (float): Assigned reps.
            sigma_L (float): Std deviation for the left side.
            sigma_R (float): Std deviation for the right side.

        Returns:
            function: Normalized PDF function.
        """
        def combined_pdf_normalized(x):
            return combined_pdf(x, r, sigma_L, sigma_R)

        normalization_constant, _ = quad(combined_pdf_normalized, -0.9 * r, 3 * r)
        return lambda x: combined_pdf_normalized(x) / normalization_constant

    # Generate random samples for each assigned rep
    sampled_vals = np.empty(size)

    for i, r_assigned in enumerate(r_assigned_array):
        x_vals = np.linspace(min_bound[i], max_bound[i], 1000)
        normalized_pdf = normalize_pdf(r_assigned, sigma_L[i], sigma_R[i])
        pdf_vals = np.array([normalized_pdf(x) for x in x_vals])
        cdf_vals = np.cumsum(pdf_vals)
        cdf_vals /= cdf_vals[-1]

        random_probs = np.random.uniform(0, 1)
        sampled_vals[i] = np.interp(random_probs, cdf_vals, x_vals)

    return sampled_vals

def simulate_actual_lift_data(assigned_weights_df, num_iterations=5):
    """
    Simulate actual lift data using vectorized operations and parallelization.

    Parameters:
        assigned_weights_df (pd.DataFrame): DataFrame with assigned weights and reps.
        num_iterations (int): Number of iterations for simulation.

    Returns:
        pd.DataFrame: Simulated lift data.
    """
    np.random.seed(42)  # For reproducibility

    def simulate_iteration(df):
        df = df.copy()
        r_assigned = df['# of Reps'].values
        w_assigned = df['Assigned Weight'].values

        # Generate rep differences and calculate actual reps
        rep_differences = generate_rep_differences_vectorized(r_assigned)
        df['Actual Reps'] = r_assigned + rep_differences

        # Generate actual weights based on a normal distribution
        df['Actual Weight'] = np.random.normal(loc=w_assigned, scale=w_assigned * 0.1)
        return df

    # Parallelize iterations
    simulated_data = Parallel(n_jobs=-1)(
        delayed(simulate_iteration)(assigned_weights_df) for _ in range(num_iterations)
    )

    # Concatenate results
    return pd.concat(simulated_data, ignore_index=True)

def analyze_simulated_data(simulated_data):
    # Analyze Rep Differences by Exercise
    grouped_reps = simulated_data.groupby('Exercise')
    num_exercises = len(grouped_reps)
    rows = int(np.ceil(np.sqrt(num_exercises)))
    cols = int(np.ceil(num_exercises / rows))

    # Create figure for all Rep Differences
    fig_reps, axs_reps = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axs_reps = axs_reps.flatten()
    fig_reps.suptitle("Rep Differences Grouped by Exercise", fontsize=16)

    for i, (exercise, group) in enumerate(grouped_reps):
        rep_diff = group['Actual Reps'] - group['# of Reps']
        axs_reps[i].hist(rep_diff, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axs_reps[i].set_title(f"Exercise: {exercise}")
        axs_reps[i].set_xlabel("Rep Differences")
        axs_reps[i].set_ylabel("Frequency")

    # Hide unused subplots for Rep Differences
    for j in range(i + 1, len(axs_reps)):
        axs_reps[j].axis('off')

    fig_reps.tight_layout(rect=[0, 0, 1, 0.95])
    fig_reps.savefig("Rep_Differences_by_Exercise.png")
    plt.close(fig_reps)

    # Analyze Weight Differences by Exercise and Player
    grouped_weights = simulated_data.groupby(['Player', 'Exercise'])

    # Generate one figure per player
    for player, player_group in simulated_data.groupby('Player'):
        player_exercises = player_group['Exercise'].unique()
        num_exercises = len(player_exercises)
        rows = int(np.ceil(np.sqrt(num_exercises)))
        cols = int(np.ceil(num_exercises / rows))

        fig_weights, axs_weights = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        axs_weights = axs_weights.flatten()
        fig_weights.suptitle(f"Weight Differences for Player: {player}", fontsize=16)

        for i, (exercise, group) in enumerate(player_group.groupby('Exercise')):
            weight_diff = group['Actual Weight'] - group['Assigned Weight']
            axs_weights[i].hist(weight_diff, bins=20, alpha=0.7, color='green', edgecolor='black')
            axs_weights[i].set_title(f"Exercise: {exercise}")
            axs_weights[i].set_xlabel("Weight Differences")
            axs_weights[i].set_ylabel("Frequency")

        # Hide unused subplots for Weight Differences
        for j in range(i + 1, len(axs_weights)):
            axs_weights[j].axis('off')

        fig_weights.tight_layout(rect=[0, 0, 1, 0.95])
        fig_weights.savefig(f"Weight_Differences_Player_{player}.png")
        plt.close(fig_weights)

def main():
    program_df = read_google_sheets('After-School Lifting', 'CompleteProgram')
    core_maxes_df = read_google_sheets('After-School Lifting', 'Maxes')

    exercise_functions = fit_functions_by_exercise(program_df)

    assigned_weights_df = calculate_assigned_weights(program_df, core_maxes_df, exercise_functions)

    simulated_data = simulate_actual_lift_data(assigned_weights_df, num_iterations=5)

    analyze_simulated_data(simulated_data)

    write_to_google_sheet('After-School Lifting', 'AssignedWeights', assigned_weights_df)
    write_to_google_sheet('After-School Lifting', 'SimulatedLiftData', simulated_data)

if __name__ == "__main__":
    main()
