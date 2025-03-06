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

def simulate_reps(assigned_reps):
    """ Simulates executed reps based on an asymmetric probability distribution. """
    
    # Define rep variation bounds
    delta_r_min = -0.9 * assigned_reps
    delta_r_max = 5.0 * assigned_reps

    # Compute standard deviations for left and right variations
    sigma_L = (0.997 * -delta_r_min) / 3
    sigma_R = (0.997 * delta_r_max) / 3

    # Generate a deviation within allowed limits
    while True:
        delta_r = np.random.normal(0, sigma_L if np.random.rand() < 0.5 else sigma_R)
        new_reps = assigned_reps + delta_r
        if delta_r_min <= new_reps - assigned_reps <= delta_r_max:
            break  # Ensure only valid values are returned

    return int(round(new_reps))  # Convert to integer reps

def simulate_weights(assigned_weight):
    """
    Simulates the weight a player actually lifts.
    
    - Almost always less than assigned weight.
    - Lifted weight falls between 60% and 150% of the assigned weight.
    """
    
    # Define the probability distribution
    if np.random.rand() < 0.95:  # 95% chance of lifting less than assigned weight
        simulated_weight = np.random.uniform(0.60, 1.00) * assigned_weight
    else:  # 5% chance of lifting above assigned weight
        simulated_weight = np.random.uniform(1.00, 1.50) * assigned_weight

    return round(simulated_weight, 2)  # Round to 2 decimal places

def run_simulation(input_data, maxes_df):
    """Simulate exercise performance based on input data."""
    # Function logic
    """ Runs the full simulation on the assigned weights dataset. """
    logging.info("🚀 Running simulation on assigned weights dataset...")

    # Filter out 'NRM' rows before simulation
    # valid_assigned_weights_df = assigned_weights_df[assigned_weights_df["Assigned Weight"] != "NRM"].copy()
    # Ensure only players from the Maxes tab are included
    active_players = assigned_weights_df["Player"].unique()  # Get players in Assigned Weights
    maxes_players = maxes_df["Player"].unique()  # Get players in Maxes tab

    # Filter assigned_weights_df to include only active players
    valid_assigned_weights_df = assigned_weights_df[
        (assigned_weights_df["Assigned Weight"] != "NRM") &
        (assigned_weights_df["Player"].isin(maxes_players))
    ]

    if valid_assigned_weights_df.empty:
        logging.warning("⚠️ No valid assigned weights available for simulation.")
        return pd.DataFrame()

    # expanded_df = valid_assigned_weights_df.loc[valid_assigned_weights_df.index.repeat(SIMULATION_ROUNDS)].copy()
    expanded_df = valid_assigned_weights_df.loc[valid_assigned_weights_df.index.repeat(SIMULATION_ROUNDS)]
    expanded_df["Simulation Round"] = np.tile(np.arange(1, SIMULATION_ROUNDS + 1), len(valid_assigned_weights_df))

    expanded_df["Simulated Reps"] = expanded_df["# of Reps"].apply(simulate_reps)
    expanded_df["Simulated Weight"] = expanded_df["Assigned Weight"].apply(simulate_weights)

    # Ensure data is properly formatted
    # logging.info(f"✅ Simulated Data Shape: {expanded_df.shape}")

    # Save locally for debugging
    simulated_data_path = os.path.join(project_root, "data/simulated_data.pkl")
    expanded_df.to_pickle(simulated_data_path)
    logging.info(f"✅ Simulated data saved to {simulated_data_path}")

    return expanded_df
