import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
# warnings.simplefilter("always")  # Show all warnings.
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
from scipy.stats import norm
from scipy.integrate import quad
import os
os.environ["OMP_NUM_THREADS"] = "1"  # Limit threads for OpenMP
os.environ["MKL_NUM_THREADS"] = "1"  # Limit threads for MKL (NumPy/Scipy backend)
from google.auth.transport.requests import AuthorizedSession
from concurrent.futures import ThreadPoolExecutor
import time

def retry(func, retries=3, delay=10):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
    raise Exception("All retry attempts failed.")

def write_chunk(sheet, worksheet_name, chunk, headers=None):
    if headers:
        sheet.update([headers] + chunk)
    else:
        sheet.append_rows(chunk)

# Increase timeout for AuthorizedSession
class CustomAuthorizedSession(AuthorizedSession):
    def request(self, *args, **kwargs):
        kwargs['timeout'] = 300  # Set timeout to 300 seconds
        return super().request(*args, **kwargs)

# Enable tracebacks for warnings.
warnings.filterwarnings("always", category=FutureWarning)

from google.oauth2.service_account import Credentials
import gspread

def authorize_google_client():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = Credentials.from_service_account_file(
        '/Users/ricky.staugustine/Documents/FB/Lifting/ntafterschoollifting-b8f7a5923646.json', scopes=scope
    )
    client = gspread.authorize(credentials)
    return client

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

def write_to_google_sheet(sheet_name, worksheet_name, data, batch_size=500):
    """
    Write data to a Google Sheet using batch updates to minimize API calls.
    """
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
    headers = data.columns.tolist()
    rows = data.values.tolist()

    # Write headers
    print(f"Writing headers to worksheet: {worksheet_name}")
    worksheet.update(range_name='A1:Z1', values=[headers])

    # Batch write data rows
    print(f"Batch updating worksheet: {worksheet_name} with batch size: {batch_size}")
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        start_row = i + 2  # Account for headers
        end_row = start_row + len(batch) - 1
        range_to_update = f"A{start_row}:Z{end_row}"
        try:
            worksheet.update(range_name=range_to_update, values=batch)
        except Exception as e:
            print(f"Error updating range {range_to_update}: {e}")

    print(f"Successfully wrote {len(rows)} rows of data.")
      
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

def calculate_assigned_weights_vectorized(program_df, core_maxes_df, exercise_functions):
    """
    Vectorized calculation of assigned weights for all players iterating through the program.
    """
    # Expand program_df for all players
    repeated_program_df = pd.concat([program_df] * len(core_maxes_df), ignore_index=True)
    repeated_program_df['Player'] = np.tile(core_maxes_df['Player'].values, len(program_df))

    # Map exercises to their relevant cores
    exercise_to_core = program_df[['Exercise', 'Relevant Core']].drop_duplicates().set_index('Exercise')['Relevant Core'].to_dict()
    repeated_program_df['Relevant Core'] = repeated_program_df['Exercise'].map(exercise_to_core)

    # Create a mapping for relevant core maxes
    core_max_mapping = core_maxes_df.set_index('Player').stack().to_dict()

    # Create a mapping of (Player, Relevant Core) to Relevant Core Max
    player_core_to_max = (
        core_maxes_df.melt(id_vars='Player', var_name='Relevant Core', value_name='Core Max')
        .dropna(subset=['Core Max'])  # Remove rows with missing maxes
        .set_index(['Player', 'Relevant Core'])['Core Max']
        .to_dict()
    )
    
    # Map Relevant Core to exercises
    exercise_to_core = program_df[['Exercise', 'Relevant Core']].drop_duplicates().set_index('Exercise')['Relevant Core']
    
    # Expand program_df for all players
    repeated_program_df = pd.concat([program_df] * len(core_maxes_df), ignore_index=True)
    repeated_program_df['Player'] = np.tile(core_maxes_df['Player'].values, len(program_df))
    
    # Map Relevant Core dynamically
    repeated_program_df['Relevant Core'] = repeated_program_df['Exercise'].map(exercise_to_core)
    
    # Use the player-core-to-max mapping to assign Relevant Core Max
    repeated_program_df['Relevant Core Max'] = repeated_program_df.set_index(['Player', 'Relevant Core']).index.map(player_core_to_max)
    
    # Drop rows where Relevant Core Max is missing or invalid
    repeated_program_df = repeated_program_df.dropna(subset=['Relevant Core Max'])
    
    # Ensure Relevant Core Max is numeric
    repeated_program_df['Relevant Core Max'] = pd.to_numeric(repeated_program_df['Relevant Core Max'], errors='coerce')
   
    # Vectorized calculation of multipliers
    weeks = repeated_program_df['Week #'].values
    sets = repeated_program_df['Set #'].values
    reps = repeated_program_df['# of Reps'].values
    codes = repeated_program_df['Code'].values

    multipliers = np.array([
        float(exercise_functions[code](w, s, r)) for code, w, s, r in zip(codes, weeks, sets, reps)
    ])

    # Calculate assigned weights and round down to nearest 5
    relevant_core_maxes = repeated_program_df['Relevant Core Max'].astype(float).values
    repeated_program_df['Assigned Weight'] = np.floor(relevant_core_maxes * multipliers / 5) * 5

    return repeated_program_df

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

from joblib import Parallel, delayed
import numpy as np

def simulate_actual_lift_data_optimized(assigned_weights_df, num_iterations=5):
    """
    Optimized simulation of actual lift data with parallel processing.
    Simulates the entire dataset for each iteration.
    """
    def simulate_iteration(data_chunk):
        # Filter out rows where Assigned Weight is 0
        data_chunk = data_chunk[data_chunk['Assigned Weight'] > 0]

        # Vectorized calculations for one iteration
        r_assigned = data_chunk['# of Reps'].values
        w_assigned = data_chunk['Assigned Weight'].values

        # Generate rep differences
        rep_differences = generate_rep_differences_vectorized(r_assigned)
        data_chunk['Actual Reps'] = r_assigned + rep_differences

        # Generate weight differences (normal distribution)
        weight_differences = np.random.normal(
            loc=0, scale=0.1 * w_assigned, size=len(w_assigned)
        )
        data_chunk['Actual Weight'] = w_assigned + weight_differences

        return data_chunk

    # Repeat the entire dataset for each iteration
    repeated_data = pd.concat([assigned_weights_df] * num_iterations, ignore_index=True)

    # Filter out rows where Assigned Weight is 0
    repeated_data = repeated_data[repeated_data['Assigned Weight'] > 0]

    # Set the multiprocessing start method.
    multiprocessing.set_start_method("spawn", force=True)

    # Simulate data in parallel
    simulated_data = Parallel(n_jobs=-1, backend="loky")(
        delayed(simulate_iteration)(repeated_data.iloc[i::num_iterations].copy())
        for i in range(num_iterations)
    )

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
        rep_diff = rep_diff.dropna()  # Remove NaN values
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
            weight_diff = weight_diff.dropna()  # Remove NaN values
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

    assigned_weights_df = calculate_assigned_weights_vectorized(program_df, core_maxes_df, exercise_functions)

    simulated_data = simulate_actual_lift_data_optimized(assigned_weights_df, num_iterations=5)

    analyze_simulated_data(simulated_data)

    write_to_google_sheet('After-School Lifting', 'AssignedWeights', assigned_weights_df, batch_size=500)
    write_to_google_sheet('After-School Lifting', 'SimulatedLiftData', simulated_data, batch_size=500)

if __name__ == "__main__":
    main()
