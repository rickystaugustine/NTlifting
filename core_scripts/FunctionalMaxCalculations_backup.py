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
from helpers.helpers import fit_single_exercise_global, simulate_iteration, generate_rep_differences_vectorized
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime

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
    # Only melt Bench, Squat, Clean, Deadlift
    core_lifts = ["Bench", "Barbell Squat", "Clean", "Hex-Bar Deadlift"]
    flattened_core_maxes_df = core_maxes_df.melt(
        id_vars=["Player"],
        value_vars=core_lifts,      # Only these four columns
        var_name="Relevant Core",   # e.g. "Bench", "Squat", etc.
        value_name="Tested Max"     # Explicitly call this the "Tested Max"
    )
    return flattened_core_maxes_df

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

# Function: Calculate Assigned Weights
# Replace calculate_assigned_weights_vectorized to handle the new return format
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

def analyze_weight_differences(simulated_data, output_dir="weight_difference_plots"):
    """
    Analyze Weight Differences grouped by Player and Exercise, and plot distributions in a grid layout.

    Parameters:
        simulated_data (pd.DataFrame): Simulated data containing 'Actual Weight' and 'Assigned Weight'.
        output_dir (str): Directory to save weight difference plots.
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Calculate Weight Differences
    simulated_data["Weight Difference"] = simulated_data["Actual Weight"] - simulated_data["Assigned Weight"]

    # Group data by Player
    grouped_by_player = simulated_data.groupby("Player")

    for player, player_data in grouped_by_player:
        # Group player's data by Exercise
        grouped_by_exercise = player_data.groupby("Exercise")

        # Determine grid layout for subplots
        num_exercises = len(grouped_by_exercise)
        rows = int(np.ceil(np.sqrt(num_exercises)))
        cols = int(np.ceil(num_exercises / rows))

        # Setup figure and axes
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10), constrained_layout=True)
        axes = np.array(axes).reshape(-1)  # Flatten for easy indexing
        for ax in axes[num_exercises:]:
            ax.axis("off")  # Turn off unused subplots

        for ax, (exercise, exercise_data) in zip(axes, grouped_by_exercise):
            # Extract Weight Differences and Assigned Weights
            weight_differences = exercise_data["Weight Difference"]
            assigned_weights = exercise_data["Assigned Weight"]

            # Plot histogram of Weight Differences
            ax.hist(weight_differences, bins=20, alpha=0.7, color="blue", edgecolor="black")
            ax.axvline(0, color="red", linestyle="--", linewidth=1.5)  # Highlight mean
            ax.set_title(f"{exercise}", fontsize=12)
            ax.set_xlabel("Weight Difference (Actual - Assigned)", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, linestyle="--", alpha=0.7)

            # Set x-limits based on W_diff bounds: [-0.5 * W_assigned, 0.5 * W_assigned]
            max_assigned = assigned_weights.max()
            ax.set_xlim([-0.5 * max_assigned, 0.5 * max_assigned])

        # Add a super title for the player
        fig.suptitle(f"Weight Difference Distributions for Player: {player}", fontsize=16, weight="bold")

        # Save the figure
        output_file = os.path.join(output_dir, f"Weight_Differences_Player_{player}.png")
        plt.savefig(output_file)
        plt.close(fig)  # Close figure to free memory
        print(f"Saved plot for Player: {player} to {output_file}")

def calculate_functional_maxes_rowlevel(simulated_data, n_data=None):
    """
    Compute a row-level 'Functional Max' for each row in simulated_data.
    Returns a new DataFrame with columns:
      - everything from simulated_data
      - 'FunctionalMax_Row' as the final x_func
    """

    grouped = simulated_data.groupby(['Player', 'Relevant Core'])

    # We'll store x_funcs in a dict so we can join later
    row_xfunc_map = {}

    def blend_with_confidence(x_raw, tested_max, gamma, max_shift_pct):
        if pd.isna(tested_max) or tested_max <= 0:
            return x_raw

        x_conf = (1 - gamma) * tested_max + gamma * x_raw

        # Dynamically adjust bounds based on tested_max
        lower_bound = (1 - max_shift_pct) * tested_max
        upper_bound = (1 + max_shift_pct) * tested_max

        # Prevent over-smoothing at zero
        if abs(x_raw - tested_max) < 0.1 * tested_max:
            return tested_max  # If close enough, return tested_max directly

        x_clamped = max(min(x_conf, upper_bound), lower_bound)
        return x_clamped

    def get_dynamic_confidence_factors(N):
        k = 0.01
        gamma_raw = 1.0 - np.exp(-k * N)
        gamma = min(gamma_raw, 0.99)
        base_pct = 0.01
        max_additional = 0.04
        max_shift_pct = base_pct + max_additional * gamma
        return gamma, max_shift_pct
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # (B) NEW: Smooth function to recalc M_actual if r_actual != r_assigned
    # ─────────────────────────────────────────────────────────────────────────
    def smooth_m_actual(M_assigned, r_asg, r_act):
        """
        Smoothly adjusts M_assigned based on rep difference.
        factor = ( (r_act - r_asg)/r_asg ) / (1 + abs((r_act - r_asg)/r_asg))
        M_actual = M_assigned * (1 + factor)

        For small |r_act - r_asg|, factor is near 0 -> minimal shift.
        For large differences, factor approaches ±1 but never infinite.
        """
        diff = r_act - r_asg
        if r_asg == 0:
            # Avoid dividing by zero
            return M_assigned

        ratio = diff / r_asg
        factor = ratio / (1 + abs(ratio))
        # e.g. if ratio=1, factor=0.5; if ratio=2, factor~0.6667
        M_new = M_assigned * (1 + factor)
        return M_new
    # ─────────────────────────────────────────────────────────────────────────

    # If you want a global N for dynamic confidence:
    if n_data is not None:
        gamma, max_shift_pct = get_dynamic_confidence_factors(n_data)
    else:
        gamma, max_shift_pct = (0.1, 0.05)

    # Main loop
    for (player, core), group_df in grouped:
        tested_max = group_df['Tested Max'].iloc[0] if 'Tested Max' in group_df.columns else np.nan

        for idx, row in group_df.iterrows():
            W_assigned = row['Assigned Weight']
            W_actual   = row['Actual Weight']
            r_assigned = row['# of Reps']
            r_actual   = row['Actual Reps']
            M_assigned = row["Multiplier"]

            # Cases 1–3 => x_raw => x_func
            # e.g.
            if (W_actual != W_assigned) and (r_actual == r_assigned):
                x_raw = W_actual / M_assigned
            elif (W_actual == W_assigned) and (r_actual != r_assigned):
                M_actual = smooth_m_actual(M_assigned, r_assigned, r_actual)
                x_raw = W_assigned / M_actual
            else:
                # case 3 with your recursive_approx logic
                def recursive_approx(W_a, r_a, M_a):
                    """
                    Modified approach that balances weight and rep differences symmetrically.
                    """
                    x_guess_1 = W_a / M_a
                    M_calc = smooth_m_actual(M_a, r_assigned, r_a)
                    x_guess_2 = W_a / M_calc

                    # Base average
                    x_new = 0.5 * (x_guess_1 + x_guess_2)

                    # Calculate differences
                    w_diff = W_a - W_assigned
                    r_diff = r_a - r_assigned

                    # Normalize differences
                    w_ratio = w_diff / W_assigned if W_assigned != 0 else 0
                    r_ratio = r_diff / r_assigned if r_assigned != 0 else 0

                    # Symmetrically adjust x_new based on combined differences
                    adjustment = 0.5 * (w_ratio + r_ratio)
                    x_new += adjustment * x_new

                    return x_new

                x_raw = recursive_approx(W_actual, r_actual, M_assigned)

            x_func = blend_with_confidence(x_raw, tested_max, gamma, max_shift_pct)
            row_xfunc_map[idx] = x_func

    # Merge the results back
    xfunc_df = pd.DataFrame({
        "RowIndex": list(row_xfunc_map.keys()),
        "FunctionalMax_Row": list(row_xfunc_map.values())
    }).set_index("RowIndex")

    rowlevel_df = simulated_data.join(xfunc_df, how="left")
    print("[DEBUG] Sample of FunctionalMax_Row before Max Difference calculation:")
    print(rowlevel_df[['Player', 'Relevant Core', 'FunctionalMax_Row', 'Tested Max']].head())

    # Calculate Max Difference
    rowlevel_df['Max Difference'] = (
        rowlevel_df['FunctionalMax_Row'] - rowlevel_df['Tested Max']
    )
    
    # Add debug log
    print("[DEBUG] Sample of Max Difference:")
    print(rowlevel_df[['Player', 'Relevant Core', 'Max Difference']].head())

    print("[DEBUG] Distribution of Max Differentials:")
    print(rowlevel_df['Max Difference'].describe())

    return rowlevel_df

def calculate_functional_maxes_aggregated(rowlevel_df):
    """
    Aggregates row-level FunctionalMax_Row by (Player, Relevant Core).
    Returns a pivoted DataFrame with 1 row per (Player, Core).
    """

    grouped = rowlevel_df.groupby(['Player', 'Relevant Core'])
    results = []

    for (player, core), group_df in grouped:
        tested_max = group_df['Tested Max'].iloc[0] if 'Tested Max' in group_df.columns else np.nan

        if 'FunctionalMax_Row' in group_df.columns:
            # e.g. median
            functional_max = group_df['FunctionalMax_Row'].median()
            diff = functional_max - tested_max if pd.notna(tested_max) else np.nan
        else:
            functional_max = np.nan
            diff = np.nan

        results.append({
            'Player': player,
            'Core': core,
            'TestedMax': tested_max,
            'FunctionalMax': functional_max,
            'Diff': diff
        })

    final_df = pd.DataFrame(results)

    final_wide = final_df.pivot(index='Player', columns='Core', values=['TestedMax', 'FunctionalMax', 'Diff'])
    final_wide['TotalTested'] = final_wide['TestedMax'].sum(axis=1)
    final_wide['TotalFunctional'] = final_wide['FunctionalMax'].sum(axis=1)
    final_wide['TotalDiff'] = final_wide['TotalFunctional'] - final_wide['TotalTested']

    final_wide.reset_index(inplace=True)
    final_wide.columns = ['Player'] + [
        f"{col[1]} {col[0]}" if col[1] else col[0]
        for col in final_wide.columns[1:]
    ]
    return final_wide

def plot_max_differentials(rowlevel_df, output_file="Differentials.png"):
    """
    Plot the max differentials grouped by core exercises and total.
    """
    grouped = rowlevel_df.groupby('Relevant Core')

    fig = make_subplots(
        rows=1,
        cols=len(grouped.groups) + 1,  # +1 for the total
        subplot_titles=list(grouped.groups.keys()) + ["Total"]
    )

    for i, (core, group) in enumerate(grouped, start=1):
        fig.add_trace(
            go.Histogram(
                x=group["Max Difference"],
                nbinsx=50,
                name=core
            ),
            row=1, col=i
        )

    # Add total differential column
    fig.add_trace(
        go.Histogram(
            x=rowlevel_df["Max Difference"],
            nbinsx=50,
            name="Total"
        ),
        row=1, col=len(grouped.groups) + 1
    )

    fig.update_layout(
        title="Row-Level Max Differentials by Core and Total",
        height=600,
        width=300 * (len(grouped.groups) + 1)
    )
    fig.write_image(output_file)
    print(f"Max differentials plot saved to {output_file}")

def calculate_confidence_metrics(simulated_data, functional_maxes, run_id=None, data_count=None):
    print(f"[DEBUG] Received data_count: {data_count}")

    weight_diff = simulated_data["Actual Weight"] - simulated_data["Assigned Weight"]
    rmse_weight = np.sqrt((weight_diff**2).mean())
    mae_weight  = np.mean(np.abs(weight_diff))

    reps_diff   = simulated_data["Actual Reps"] - simulated_data["# of Reps"]
    rmse_reps   = np.sqrt((reps_diff**2).mean())
    mae_reps    = np.mean(np.abs(reps_diff))

    diff_cols = [col for col in functional_maxes.columns if "Diff" in col]
    if diff_cols:
        diffs = []
        for c in diff_cols:
            diffs.extend(functional_maxes[c].dropna().values)
        diffs = np.array(diffs)
        rmse_func_tested = np.sqrt((diffs**2).mean())
        mae_func_tested  = np.mean(np.abs(diffs))
    else:
        rmse_func_tested = np.nan
        mae_func_tested  = np.nan

    print("\n[DEBUG] ----- Confidence Metrics -----")
    print(f"RMSE(Weight) = {rmse_weight:.2f}, MAE(Weight) = {mae_weight:.2f}")
    print(f"RMSE(Reps)   = {rmse_reps:.2f}, MAE(Reps)   = {mae_reps:.2f}")
    print(f"RMSE(Func-Tested) = {rmse_func_tested:.2f}, MAE(Func-Tested) = {mae_func_tested:.2f}")
    print("[DEBUG] --------------------------------\n")

    df_row = pd.DataFrame([{
        "run_id": run_id if run_id is not None else pd.Timestamp.now(),
        "data_count": data_count if data_count else 0,
        "rmse_weight": rmse_weight,
        "mae_weight": mae_weight,
        "rmse_reps": rmse_reps,
        "mae_reps": mae_reps,
        "rmse_func_tested": rmse_func_tested,
        "mae_func_tested": mae_func_tested
    }])
    return df_row

# NEW function to plot historical metrics
def plot_metrics_history(csv_file="metrics_history.csv", output_file="MetricsHistory.png"):
    """
    Plot historical error metrics (RMSE/MAE) and annotate number of data points per run.
    """

    if not os.path.exists(csv_file):
        print(f"[DEBUG] No {csv_file} found, skipping metrics history plot.")
        return

    df = pd.read_csv(csv_file)
    if df.empty:
        print(f"[DEBUG] {csv_file} is empty, skipping plot.")
        return

    # Identify columns to plot
    metrics_cols = ["rmse_weight", "mae_weight", "rmse_reps", "mae_reps", "rmse_func_tested", "mae_func_tested"]
    num_metrics = len(metrics_cols)

    # Create subplots
    fig = make_subplots(rows=1, cols=num_metrics, subplot_titles=metrics_cols)

    for i, col in enumerate(metrics_cols, start=1):
        # Plot the metric
        fig.add_trace(
            go.Scatter(
                x=df["run_id"],
                y=df[col],
                mode='lines+markers',
                name=col
            ),
            row=1, col=i
        )

        # Annotate the number of data points
        fig.add_trace(
            go.Scatter(
                x=df["run_id"],
                y=df[col],
                mode='text',
                text=df["data_count"],
                textposition='top center',
                showlegend=False
            ),
            row=1, col=i
        )

    fig.update_layout(
        title="Metrics Over Runs (Convergence/Divergence of Functional Max)",
        height=600,
        width=300 * num_metrics
    )

    # Save the plot
    fig.write_image(output_file)
    print(f"[DEBUG] Metrics history plot saved to {output_file}")

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

    # Step 4: Fit exercise functions
    print("Fitting functions by exercise...")
    exercise_functions = fit_functions_by_exercise(program_df)

    # Debugging: Ensure exercise_functions are populated
    if not exercise_functions:
        print("WARNING: No exercise functions generated!")
    else:
        print(f"Exercise Functions Keys: {list(exercise_functions.keys())[:5]}")  # Log a sample of the keys

    # Step 5: Calculate multipliers and assigned weights
    print("Calculating multipliers and assigned weights...")
    repeated_program_df = calculate_assigned_weights_vectorized(
        repeated_program_df, flattened_core_maxes_df, exercise_functions
    )
    print(f"Assigned Weights (Sample Rows):\n{repeated_program_df[['Player', 'Exercise', 'Assigned Weight']].head()}")

    # Step 6: Simulate actual lift data
    simulated_data = simulate_actual_lift_data_optimized(repeated_program_df, num_iterations=5)

    # (A) Grab N_total if you want it
    N_total = len(simulated_data)
    print(f"[DEBUG] N_total (Number of data points): {N_total}")

    # (B) Row-level
    print("Calculating row-level functional maxes...")
    rowlevel_df = calculate_functional_maxes_rowlevel(simulated_data, n_data=N_total)

    # (C) Plot a big histogram of the row-level data
#    import plotly.express as px
#    fig = px.histogram(
#        rowlevel_df, 
#        x="FunctionalMax_Row", 
#        nbins=50, 
#        title="Row-Level Functional Maxes"
#    )
#    fig.write_image("AllFunctionalMaxes.png")
#    print("Saved row-level functional max distribution to AllFunctionalMaxes.png")

    # (D) Now produce the aggregated result
    print("Aggregating functional maxes by player/core...")
    try:
        functional_maxes = calculate_functional_maxes_aggregated(rowlevel_df)
    except Exception as e:
        print(f"Error calculating functional maxes: {e}")
        return
    print(f"Aggregated Functional Maxes (Sample Rows):\n{functional_maxes.head()}")

    # Step 7: Analyze simulated data (existing code)
    print("Analyzing simulated data...")
    analyze_simulated_data(simulated_data)

    # Step 8: Analyze Weight Differences
    print("Analyzing weight differences...")
    analyze_weight_differences(simulated_data, output_dir="WeightDifferencePlots")

    # Step 9: Plot the aggregated differentials
    print("Plotting max differentials...")
    plot_max_differentials(rowlevel_df, output_file="Differentials.png")

    # Step 10: gather confidence metrics
    import pandas as pd
    metrics_row = calculate_confidence_metrics(
        simulated_data=simulated_data,
        functional_maxes=functional_maxes,
        run_id=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        data_count=N_total
    )
    # Append to CSV
    metrics_file = "metrics_history.csv"

    # Debugging the new metrics_row
    print("[DEBUG] New metrics_row to be added:")
    print(metrics_row)

    if os.path.exists(metrics_file):
        existing = pd.read_csv(metrics_file)
        print("[DEBUG] Existing metrics history:")
        print(existing.tail())  # Check the last few rows of the existing metrics history

        updated = pd.concat([existing, metrics_row], ignore_index=True)
    else:
        print("[DEBUG] No existing metrics history found. Creating a new file.")
        updated = metrics_row

    # Debugging the combined DataFrame before saving
    print("[DEBUG] Combined metrics history to be written:")
    print(updated.tail())

    # Save to CSV
    updated.to_csv(metrics_file, index=False)
    print(f"[DEBUG] Wrote updated metrics to {metrics_file}")

    # Plot the history
    plot_metrics_history(csv_file=metrics_file, output_file="MetricsHistory.png")

    # Step 11: Write results to Google Sheets
    print("Writing results to Google Sheets...")
    write_to_google_sheet("After-School Lifting", "AssignedWeights", repeated_program_df)
    write_to_google_sheet("After-School Lifting", "SimulatedLiftData", simulated_data)
    write_to_google_sheet("After-School Lifting", "FunctionalMaxes", functional_maxes)
    print("Process completed successfully!")

if __name__ == "__main__":
    main()
