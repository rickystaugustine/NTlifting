import pandas as pd
import numpy as np
import logging
import sys
import os
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load the multiplier fits directly from the .pkl file
def load_multiplier_fits():
    """Loads the multiplier_fits.pkl file from the explicitly defined path."""
    pkl_path = "/Users/ricky.staugustine/Documents/Programming/GitHub/NTlifting/execution/helpers/multiplier_fits.pkl"  # ✅ Explicit correct path

    try:
        with open(pkl_path, "rb") as file:
            multiplier_fits = pickle.load(file)
        logging.info(f"✅ Successfully loaded multiplier_fits.pkl from {pkl_path}")
        return multiplier_fits
    except Exception as e:
        logging.error(f"❌ Error loading multiplier_fits.pkl from {pkl_path}: {e}")
        return {}

multiplier_fits = load_multiplier_fits()  # Load coefficients at the start

if not multiplier_fits:
    logging.error("❌ Critical Error: multiplier_fits.pkl is missing. Ensure it is generated before running this script.")
    sys.exit(1)  # Exit the program if the file is missing

categorized_exercises = set()  # Store already categorized exercises

def is_constant_multiplier(exercise_code, multiplier_fits):
    """Detects if an exercise is a constant multiplier by checking its stored coefficients using Exercise Code."""
    try:
        if exercise_code not in multiplier_fits:
            logging.warning(f"⚠️ Warning: Exercise Code '{exercise_code}' not found in multiplier_fits.")
            return False  # Assume it's a fitted multiplier if not found

        coeffs = multiplier_fits[exercise_code]

        # Check if first 3 coefficients are insignificant
        nonzero_coeffs = [c for c in coeffs[:3] if abs(c) > 1e-6]

        is_constant = len(nonzero_coeffs) == 0
        return is_constant

    except Exception as e:
        logging.warning(f"⚠️ Error checking constant multiplier for Exercise Code '{exercise_code}': {e}")

    return False

def count_matching_reps(expanded_df):
    """Counts how many simulation instances have Simulated Reps equal to Assigned Reps."""
    expanded_df["Reps Match"] = expanded_df.apply(lambda row: int(row["Simulated Reps"]) == int(row["# of Reps"]), axis=1)
    matching_count = expanded_df["Reps Match"].sum()
    logging.info(f"✅ Total instances where r_ac == r_as: {matching_count} out of {len(expanded_df)} total simulations.")
    return matching_count

def calculate_functional_max(row):
    """Calculates Functional Max based on Method."""
    if row["Method"] == "Ratio":
        if row["Tested Max"] > 0 and row["Assigned Weight"] > 0:
            return row["Tested Max"] * (row["Simulated Weight"] / row["Assigned Weight"])
        else:
            return None  # Prevents division errors
    return None  # Placeholder for other methods

def assign_cases(expanded_df):
    """Assigns Cases, Multiplier Type, Method, and Adjusted Multiplier."""
    expanded_df["Case"] = 3  # Default all to Case 3
    expanded_df["Code"] = expanded_df["Code"].astype(int)  # ✅ Convert to integer before lookup
    expanded_df["Multiplier Type"] = "Fitted"  # Default all to Fitted
    expanded_df["Method"] = "Iterative"  # Default all to Iterative
    expanded_df["Adjusted Multiplier"] = None  # Placeholder for Adjusted Multiplier

    expanded_df = expanded_df.reset_index(drop=True)
    # Optional: Explicitly cast weight columns to np.float64 for precision
    expanded_df["Simulated Weight"] = expanded_df["Simulated Weight"].astype(np.float64)
    expanded_df["Assigned Weight"] = expanded_df["Assigned Weight"].astype(np.float64)

    # Calculate Reps Match and Weights Close before assigning cases
    expanded_df["Reps Match"] = expanded_df.apply(
        lambda row: int(row["Simulated Reps"]) == int(row["# of Reps"]), axis=1
    )
    expanded_df["Weights Close"] = expanded_df.apply(
        lambda row: abs(float(row["Simulated Weight"]) - float(row["Assigned Weight"])) < 1, axis=1
    )

    expanded_df.loc[(expanded_df["Reps Match"]) & (~expanded_df["Weights Close"]), "Case"] = 1
    expanded_df.loc[(~expanded_df["Reps Match"]) & (expanded_df["Weights Close"]), "Case"] = 2

    # Assign "Method" Based on Case and Multiplier Type
    expanded_df.loc[expanded_df["Case"] == 1, "Method"] = "Ratio"
    expanded_df.loc[(expanded_df["Case"] == 2) & (expanded_df["Multiplier Type"] == "Constant"), "Method"] = "Ratio"
    expanded_df.loc[(expanded_df["Case"] == 2) & (expanded_df["Multiplier Type"] == "Fitted"), "Method"] = "Function"

    # Determine if the exercise uses a Constant or Fitted multiplier using Exercise Code
    expanded_df["Multiplier Type"] = expanded_df["Code"].apply(lambda code: "Constant" if is_constant_multiplier(code, multiplier_fits) else "Fitted")

    # Ensure "Multiplier of Max" Column Exists Before Assigning Adjusted Multiplier
    if "Multiplier of Max" in expanded_df.columns:
        # Assign "Adjusted Multiplier" Based on Method
        mask_ratio = expanded_df["Method"] == "Ratio"
        expanded_df.loc[mask_ratio, "Adjusted Multiplier"] = expanded_df.loc[mask_ratio, "Multiplier of Max"].values

        mask_function = expanded_df["Method"] == "Function"
        expanded_df.loc[mask_function, "Adjusted Multiplier"] = (
            expanded_df.loc[mask_function, "Tested Max"].values /
            expanded_df.loc[mask_function, "Simulated Weight"].values
        )
    else:
        logging.error("❌ ERROR: 'Multiplier of Max' column is missing from expanded_df.")

    # Calculate "Functional Max" Based on Method
    expanded_df.loc[expanded_df["Method"] == "Ratio", "Functional Max"] = (
        expanded_df["Tested Max"].astype(np.float64) * 
        (expanded_df["Simulated Weight"].astype(np.float64) / expanded_df["Assigned Weight"].astype(np.float64))
    )
    expanded_df.loc[expanded_df["Method"] == "Function", "Functional Max"] = (
        expanded_df["Simulated Weight"].astype(np.float64) * 
        expanded_df["Adjusted Multiplier"].astype(np.float64)
    )
    expanded_df.loc[expanded_df["Method"] == "Iterative", "Functional Max"] = 0

    # Convert to numeric, fill missing values, and explicitly cast to np.float64
    expanded_df["Adjusted Multiplier"] = pd.to_numeric(expanded_df["Adjusted Multiplier"], errors="coerce").fillna(0).astype(np.float64)
    expanded_df["Functional Max"] = pd.to_numeric(expanded_df["Functional Max"], errors="coerce").fillna(0).astype(np.float64)

    # Ensure that Functional Max is not missing before upload
    if "Functional Max" not in expanded_df.columns:
        logging.error("❌ ERROR: 'Functional Max' column is missing from expanded_df after assignment.")

    expanded_df["Functional Max"] = expanded_df["Functional Max"].astype(np.float64)

    return expanded_df
