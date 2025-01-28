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

SIMULATION_ROUNDS = 5

def simulate_reps(reps):
    """ Simulate repetitions using a normal distribution around the assigned reps. """
    return np.round(norm.rvs(loc=reps, scale=1, size=SIMULATION_ROUNDS)).astype(int)

def simulate_weights(weight):
    """ Simulate weight variations using a normal distribution around the assigned weight. """
    return np.round(norm.rvs(loc=weight, scale=2, size=SIMULATION_ROUNDS), 1)

def run_simulation(input_data):
    """Simulate exercise performance based on input data."""
    # Function logic
    """ Runs the full simulation on the assigned weights dataset. """
    logging.info("🚀 Running simulation on assigned weights dataset...")

    # Filter out 'NRM' rows before simulation
    valid_assigned_weights_df = assigned_weights_df[assigned_weights_df["Assigned Weight"] != "NRM"].copy()
    
    if valid_assigned_weights_df.empty:
        logging.warning("⚠️ No valid assigned weights available for simulation.")
        return pd.DataFrame()

    expanded_df = valid_assigned_weights_df.loc[valid_assigned_weights_df.index.repeat(SIMULATION_ROUNDS)].copy()
    expanded_df["Simulation Round"] = np.tile(np.arange(1, SIMULATION_ROUNDS + 1), len(valid_assigned_weights_df))

    expanded_df["Simulated Reps"] = valid_assigned_weights_df["# of Reps"].apply(simulate_reps).explode().values
    expanded_df["Simulated Weight"] = valid_assigned_weights_df["Assigned Weight"].apply(simulate_weights).explode().values

    # Ensure data is properly formatted
    logging.info(f"✅ Simulated Data Shape: {expanded_df.shape}")

    # Save locally for debugging
    simulated_data_path = os.path.join(project_root, "data/simulated_data.pkl")
    expanded_df.to_pickle(simulated_data_path)
    logging.info(f"✅ Simulated data saved to {simulated_data_path}")

    return expanded_df
