import numpy as np
import pandas as pd
import logging
import os
import sys
from scipy.stats import norm
from execution.helpers.google_sheets_utils import write_to_google_sheet

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load assigned weights data
assigned_weights_path = os.path.join(project_root, "data/assigned_weights.pkl")
if not os.path.exists(assigned_weights_path):
    raise FileNotFoundError(f"❌ ERROR: Assigned weights data file is missing at {assigned_weights_path}")

assigned_weights_df = pd.read_pickle(assigned_weights_path)

# Ensure required columns exist
required_columns = {"Player", "Exercise", "Assigned Weight", "# of Reps"}
if not required_columns.issubset(assigned_weights_df.columns):
    raise ValueError(f"❌ ERROR: Missing required columns in assigned_weights_df: {required_columns - set(assigned_weights_df.columns)}")

# Clean dataset
assigned_weights_df = assigned_weights_df[assigned_weights_df["Assigned Weight"] != "NRM"].copy()
assigned_weights_df["Assigned Weight"] = pd.to_numeric(assigned_weights_df["Assigned Weight"], errors="coerce")
assigned_weights_df["# of Reps"] = pd.to_numeric(assigned_weights_df["# of Reps"], errors="coerce")

# Simulation parameters
SIMULATION_ROUNDS = 5

# Simulate actual reps executed
def simulate_reps(assigned_reps, n=SIMULATION_ROUNDS):
    delta_r_min, delta_r_max = -0.9 * assigned_reps, 5.0 * assigned_reps
    sigma_l, sigma_r = (0.997 * abs(delta_r_min)) / 3, (0.997 * abs(delta_r_max)) / 3
    rand_values = np.random.uniform(0, 1, size=(len(assigned_reps), n))
    left_dist, right_dist = norm(loc=0, scale=sigma_l[:, None]), norm(loc=0, scale=sigma_r[:, None])
    deltas = np.where(rand_values < 0.5, left_dist.ppf(rand_values), right_dist.ppf(rand_values))
    return np.clip(assigned_reps[:, None] + deltas, 0, None)

# Simulate actual weight lifted
def simulate_weights(assigned_weight, n=SIMULATION_ROUNDS):
    sigma = 0.1 * assigned_weight
    return np.clip(np.random.normal(assigned_weight[:, None], sigma[:, None], size=(len(assigned_weight), n)), 
                   0.5 * assigned_weight[:, None], 1.5 * assigned_weight[:, None])

def simulate_iteration(row):
    assigned_reps = row["# of Reps"]
    return simulate_reps(np.array([assigned_reps]))[0]  # Single simulation round

def run_simulation(data=None):
    """ Runs the full simulation on the assigned weights dataset. """
    logging.info("🚀 Running simulation on assigned weights dataset...")

    # Ensure data is a DataFrame
    if isinstance(data, dict):
        logging.warning("⚠️ Data passed as a dictionary. Converting to DataFrame...")
        data = pd.DataFrame([data])

    df = data if isinstance(data, pd.DataFrame) else assigned_weights_df
    df.rename(columns={"Reps": "# of Reps", "Num Reps": "# of Reps"}, inplace=True)

    if "# of Reps" not in df.columns:
        logging.error("❌ ERROR: '# of Reps' column is missing! Assigning default value (5).")
        df["# of Reps"] = 5

    df["Simulated Reps"] = df.apply(simulate_iteration, axis=1)

    # ✅ Removed duplicate pickle save

    # ✅ Apply transformation inline instead of using a separate `df_cleaned` variable
    write_to_google_sheet(
        "After-School Lifting", 
        "SimulatedData", 
        df.applymap(lambda x: ", ".join(map(str, x)) if isinstance(x, (np.ndarray, list)) else x)
    )
    logging.info("✅ Simulated data successfully uploaded to Google Sheets!")

    return df.to_dict(orient="records")

# Expand dataframe for simulation
expanded_df = assigned_weights_df.loc[assigned_weights_df.index.repeat(SIMULATION_ROUNDS)].copy()
expanded_df["Simulation Round"] = np.tile(np.arange(1, SIMULATION_ROUNDS + 1), len(assigned_weights_df))
expanded_df["Simulated Weight"] = simulate_weights(assigned_weights_df["Assigned Weight"].values).flatten()
expanded_df["Simulated Reps"] = simulate_reps(assigned_weights_df["# of Reps"].values).flatten()

# ✅ Save expanded simulated data only once
simulated_data_path = os.path.join(project_root, "data/simulated_data.pkl")
expanded_df.to_pickle(simulated_data_path)
logging.info(f"✅ Simulated data saved to {simulated_data_path}")
