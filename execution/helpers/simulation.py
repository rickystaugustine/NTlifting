import numpy as np
import pandas as pd
import logging
import os
import sys
import time
from scipy.stats import norm
from execution.helpers.google_sheets_utils import write_to_google_sheet
from execution.helpers.functional_max_calc import calculate_functional_max

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

    assigned_reps = int(assigned_reps)  # Ensure it's an integer.
    delta_r_min = -0.9 * assigned_reps
    delta_r_max = 5.0 * assigned_reps

    # Compute standard deviations for left and right variations
    sigma_L = (0.997 * -delta_r_min) / 3
    sigma_R = (0.997 * delta_r_max) / 3

    # Generate a deviation within allowed limits
    while True:
        delta_r = np.random.normal(0, sigma_L if np.random.rand() < 0.5 else sigma_R)
        new_reps = max(1, assigned_reps + delta_r)  # Ensure reps are at least 1

        if delta_r_min <= new_reps - assigned_reps <= delta_r_max:
            break  # Ensure only valid values are returned

    return int(round(new_reps))  # Convert to integer reps

def simulate_weights(assigned_weight):
    """
    Simulates the weight a player actually lifts.
    
    - Almost always less than assigned weight.
    - Lifted weight falls between 60% and 150% of the assigned weight.
    """
    assigned_weight = float(assigned_weight)  # Ensure it's a float

    # Define the probability distribution
    if np.random.rand() < 0.95:  # 95% chance of lifting less than assigned weight
        simulated_weight = np.random.uniform(0.60, 1.00) * assigned_weight
    else:  # 5% chance of lifting above assigned weight
        simulated_weight = np.random.uniform(1.00, 1.50) * assigned_weight

    return round(simulated_weight, 2)  # Round to 2 decimal places

def run_simulation(input_data, maxes_df, constant_multiplier_exercises):
    """Runs the full simulation on the assigned weights dataset."""
    # logging.info("🚀 Running simulation on assigned weights dataset...")

    step_start_time = time.time()

    # Ensure only players from the Maxes tab are included
    active_players = input_data["Player"].unique()
    maxes_players = maxes_df["Player"].unique()

    # Filter input_data to include only active players
    valid_assigned_weights_df = input_data[
        (input_data["Assigned Weight"] != "NRM") &
        (input_data["Player"].isin(maxes_players))
    ]

    if valid_assigned_weights_df.empty:
        # logging.warning("⚠️ No valid assigned weights available for simulation.")
        return pd.DataFrame()

    # Expand dataset for multiple simulation rounds
    expanded_df = valid_assigned_weights_df.loc[valid_assigned_weights_df.index.repeat(SIMULATION_ROUNDS)]
    expanded_df["Simulation Round"] = np.tile(np.arange(1, SIMULATION_ROUNDS + 1), len(valid_assigned_weights_df))

    # Simulate reps and weights
    expanded_df["Simulated Reps"] = expanded_df["# of Reps"].apply(simulate_reps)
    expanded_df["Simulated Weight"] = expanded_df["Assigned Weight"].apply(simulate_weights)

    # Convert all relevant columns to numeric values
    expanded_df["Simulated Weight"] = pd.to_numeric(expanded_df["Simulated Weight"], errors="coerce")
    expanded_df["Assigned Weight"] = pd.to_numeric(expanded_df["Assigned Weight"], errors="coerce")
    expanded_df["Tested Max"] = pd.to_numeric(expanded_df["Tested Max"], errors="coerce")
    expanded_df["Multiplier of Max"] = pd.to_numeric(expanded_df["Multiplier of Max"], errors="coerce")

    # Classify each row into one of three cases
    def classify_case(row):
        weight_tolerance = 1.0  # Allow small floating-point differences
        sim_weight = float(row["Simulated Weight"])
        assigned_weight = float(row["Assigned Weight"])
        sim_reps = int(row["Simulated Reps"])
        assigned_reps = int(row["# of Reps"])

        if sim_reps == assigned_reps and abs(sim_weight - assigned_weight) > weight_tolerance:
            return 1  # Case 1: r_ac = r_as & W_ac ≠ W_as
        elif sim_reps != assigned_reps and abs(sim_weight - assigned_weight) <= weight_tolerance:
            return 2  # Case 2: r_ac ≠ r_as & W_ac = W_as
        else:
            return 3  # Case 3: r_ac ≠ r_as & W_ac ≠ W_as

    expanded_df["Case"] = expanded_df.apply(classify_case, axis=1)

    # Assign "Case Subcategory" using passed classification data
    def classify_subcategory(row):
        if row["Case"] != 2:
            return "NRM"

        exercise_name = row["Exercise"].strip().upper()  # Normalize formatting

        # Attempt lookup using both name and index (if constant_multiplier_exercises keys are stored as numbers)
        classification = constant_multiplier_exercises.get(exercise_name)

        # If still not found, try converting to int (if the key is stored as an index)
        if classification is None and exercise_name.isdigit():
            classification = constant_multiplier_exercises.get(int(exercise_name))

        return classification if classification in ["2a"] else "NRM"  # Exclude Case 2b

    expanded_df["Case Subcategory"] = expanded_df.apply(classify_subcategory, axis=1)

    # Apply Functional Max calculation
    def calculate_functional_max(row):
        if pd.isna(row["Tested Max"]) or pd.isna(row["Simulated Weight"]) or pd.isna(row["Assigned Weight"]):
            # logging.warning(f"⚠️ NaN encountered in Functional Max calculation for row: {row}")
            return "NRM"
        
        if row["Case"] == 1:
            return (row["Simulated Weight"] / row["Assigned Weight"]) * row["Tested Max"]

        elif row["Case"] == 2 and row["Case Subcategory"] == "2a":
            sim_reps = int(row["Simulated Reps"])
            assigned_reps = int(row["# of Reps"])

            if sim_reps != assigned_reps and row["Tested Max"] > 0:  # Ensuring M_ac is correctly scaled
                M_ac = max(float(row["Multiplier of Max"]) * (assigned_reps / sim_reps), 0.1)  # Ensures M_ac is never too small
                if M_ac == 0:
                    # logging.warning(f"⚠️ M_ac is zero for Exercise={row['Exercise']}, skipping Functional Max calculation.")
                    return "NRM"
                X_f = row["Simulated Weight"] / M_ac
                
                if pd.isna(X_f):
                    # logging.warning(f"⚠️ NaN encountered in Functional Max calculation -> Exercise={row['Exercise']}, M_ac={M_ac}, W_ac={row['Simulated Weight']}, X_f={X_f}")
                    return "NRM"
                
                if X_f > 1000:  # Cap unrealistic values
                    return "NRM"
                
                return X_f
            
        return "NRM"

    expanded_df["Functional Max"] = expanded_df.apply(calculate_functional_max, axis=1)

    # Convert "Functional Max" to numeric
    expanded_df["Functional Max"] = pd.to_numeric(expanded_df["Functional Max"], errors="coerce").fillna(0)

    # Remove columns that may cause JSON compliance errors when writing to Google Sheets
    expanded_df.drop(columns=["Updated Functional Max"], errors="ignore", inplace=True)

    # Ensure the entire DataFrame has no NaN values before export
    expanded_df.fillna(0, inplace=True)

    # Step tracking log for Functional Max calculation completion
    step_time = time.time() - step_start_time
    # logging.info(f"✅ Functional Max step completed in {step_time:.2f} seconds.")

    # Save locally for debugging
    simulated_data_path = os.path.join(project_root, "data/simulated_data.pkl")
    expanded_df.to_pickle(simulated_data_path)

    return expanded_df
