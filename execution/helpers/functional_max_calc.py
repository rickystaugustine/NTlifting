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
        # logging.info(f"✅ Successfully loaded multiplier_fits.pkl from {pkl_path}")
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
            # logging.warning(f"⚠️ Warning: Exercise Code '{exercise_code}' not found in multiplier_fits.")
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
    # logging.info(f"✅ Total instances where r_ac == r_as: {matching_count} out of {len(expanded_df)} total simulations.")
    return matching_count

def calculate_functional_max(row):
    """Calculates Functional Max based on Method."""
    if row["Method"] == "Ratio":
        if row["Tested Max"] > 0 and row["Assigned Weight"] > 0:
            return row["Tested Max"] * (row["Simulated Weight"] / row["Assigned Weight"])
        else:
            return None  # Prevents division errors
    return None  # Placeholder for other methods

def calculate_adjusted_multiplier_function(row, k=10):
    exercise_code = row["Code"]

    # Explicit type conversions
    try:
        week_num = float(row["Week #"])
        set_num = float(row["Set #"])
        sim_reps = float(row["Simulated Reps"])
    except Exception as e:
        logging.error(f"❌ Error converting row values to float for exercise code {exercise_code}: {e}")
        return 0

    coeffs = multiplier_fits.get(exercise_code)
    if coeffs is None:
        # logging.warning(f"⚠️ No coefficients found for exercise code {exercise_code}. Defaulting to 0.")
        return 0

    A, B, C, D = coeffs[:4]  # Extract coefficients

    try:
        adj_multiplier = (
            A * week_num +
            B * set_num +
            C * np.log(k / (sim_reps + 1)) +
            D
        )
        return adj_multiplier
    except Exception as e:
        logging.error(f"❌ Error calculating Adjusted Multiplier for exercise code {exercise_code}: {e}")
        return 0

def calculate_adjusted_multiplier_iterative(row, k=10):
    exercise_code = row["Code"]

    try:
        week_num = float(row["Week #"])
        set_num = float(row["Set #"])
        sim_reps = float(row["Simulated Reps"])
        sim_weight = float(row["Simulated Weight"])
        assigned_reps = float(row["# of Reps"])
        assigned_weight = float(row["Assigned Weight"])
        tested_max = float(row["Tested Max"])
    except Exception as e:
        logging.error(f"❌ Error converting row values for exercise code {exercise_code}: {e}")
        return 0

    coeffs = multiplier_fits.get(exercise_code)
    if coeffs is None:
        logging.error(f"❌ No coefficients found for exercise code {exercise_code}.")
        return 0

    A, B, C, D = coeffs[:4]

    try:
        expected_multiplier = (
            A * week_num +
            B * set_num +
            C * np.log(k / (sim_reps + 1)) +
            D
        )
    except Exception as e:
        logging.error(f"❌ Error calculating expected multiplier for exercise code {exercise_code}: {e}")
        return 0

    if tested_max == 0:
        logging.error(f"❌ Tested Max is zero for exercise code {exercise_code}. Cannot calculate expected weight.")
        return 0


    # Minimum expected multiplier floor calculation
    # If simulated weights can be up to 150% of assigned weight, we expect the model to account for this by providing a minimum expected weight floor
    max_sim_weight_factor = 1.5  # 150%
    min_expected_multiplier = (assigned_weight * 0.75) / tested_max if tested_max > 0 else 0
    # The 0.75 factor provides a safety margin (inverse of 150% max sim weight)

    # Apply the floor to the expected multiplier
    expected_multiplier = max(expected_multiplier, min_expected_multiplier)

    # Apply scaling factor to better align expected weights with simulation variance
    expected_weight = expected_multiplier * tested_max

    if expected_weight == 0:
        logging.error(f"❌ Expected weight is zero for exercise code {exercise_code}. Cannot calculate adjusted multiplier.")
        return 0

    adj_multiplier = sim_weight / expected_weight

    # Debug log for investigation if Adjusted Multiplier > 1
    if adj_multiplier > 1:
        logging.warning(
            f"⚠️ High Adjusted Multiplier: {adj_multiplier:.3f} "
            f"| Exercise Code: {exercise_code} "
            f"| Sim Weight: {sim_weight:.2f} "
            f"| Expected Weight: {expected_weight:.2f} "
            f"| Tested Max: {tested_max:.2f} "
            f"| Assigned Weight: {assigned_weight:.2f} "
            f"| Week: {week_num} | Set: {set_num} | Reps: {sim_reps}"
        )

    return adj_multiplier

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

    # Determine if the exercise uses a Constant or Fitted multiplier using Exercise Code
    expanded_df["Multiplier Type"] = expanded_df["Code"].apply(lambda code: "Constant" if is_constant_multiplier(code, multiplier_fits) else "Fitted")

    # Assign "Method" Based on Case and Multiplier Type
    expanded_df.loc[expanded_df["Case"] == 1, "Method"] = "Ratio"
    expanded_df.loc[(expanded_df["Case"] == 2) & (expanded_df["Multiplier Type"] == "Constant"), "Method"] = "Scale"
    expanded_df.loc[(expanded_df["Case"] == 2) & (expanded_df["Multiplier Type"] == "Fitted"), "Method"] = "Function"

    # Ensure "Multiplier of Max" Column Exists Before Assigning Adjusted Multiplier
    if "Multiplier of Max" in expanded_df.columns:
        # Assign "Adjusted Multiplier" Based on Method
        mask_ratio = expanded_df["Method"] == "Ratio"
        expanded_df.loc[mask_ratio, "Adjusted Multiplier"] = expanded_df.loc[mask_ratio, "Multiplier of Max"].values

        mask_function = expanded_df["Method"] == "Function"
        expanded_df.loc[mask_function, "Adjusted Multiplier"] = expanded_df[mask_function].apply(
            lambda row: calculate_adjusted_multiplier_function(row, k=10),  # Adjust k if needed
            axis=1
        )

        # Ensure numeric types for reps columns to prevent TypeErrors
        expanded_df["# of Reps"] = pd.to_numeric(expanded_df["# of Reps"], errors="coerce").fillna(0)
        expanded_df["Simulated Reps"] = pd.to_numeric(expanded_df["Simulated Reps"], errors="coerce").fillna(1)
        expanded_df["Simulated Reps"] = expanded_df["Simulated Reps"].replace(0, 1)

        mask_scale = expanded_df["Method"] == "Scale"
        expanded_df.loc[mask_scale, "Adjusted Multiplier"] = (
            expanded_df.loc[mask_scale, "Multiplier of Max"].values *
            (expanded_df.loc[mask_scale, "# of Reps"].values /
             expanded_df.loc[mask_scale, "Simulated Reps"].values)
        )

        mask_iterative = expanded_df["Method"] == "Iterative"
        expanded_df.loc[mask_iterative, "Adjusted Multiplier"] = expanded_df[mask_iterative].apply(
            lambda row: calculate_adjusted_multiplier_iterative(row, k=10),
            axis=1
        )
    else:
        logging.error("❌ ERROR: 'Multiplier of Max' column is missing from expanded_df.")

    # Calculate "Functional Max" Based on Method
    expanded_df.loc[expanded_df["Method"] == "Ratio", "Functional Max"] = (
        expanded_df["Tested Max"].astype(np.float64) *
        (expanded_df["Simulated Weight"].astype(np.float64) / expanded_df["Assigned Weight"].astype(np.float64))
    )
    expanded_df.loc[expanded_df["Method"] == "Function", "Functional Max"] = (
        expanded_df["Simulated Weight"].astype(np.float64) /
        expanded_df["Adjusted Multiplier"].astype(np.float64)
    )
    expanded_df.loc[expanded_df["Method"] == "Scale", "Functional Max"] = (
        expanded_df.loc[expanded_df["Method"] == "Scale", "Simulated Weight"].astype(np.float64) /
        expanded_df.loc[expanded_df["Method"] == "Scale", "Adjusted Multiplier"].astype(np.float64)
    )
    expanded_df.loc[expanded_df["Method"] == "Iterative", "Functional Max"] = (
        expanded_df.loc[expanded_df["Method"] == "Iterative", "Simulated Weight"].astype(np.float64) /
        expanded_df.loc[expanded_df["Method"] == "Iterative", "Adjusted Multiplier"].astype(np.float64)
    )

    # Convert to numeric, fill missing values, and explicitly cast to np.float64
    expanded_df["Adjusted Multiplier"] = pd.to_numeric(expanded_df["Adjusted Multiplier"], errors="coerce").fillna(0).astype(np.float64)
    expanded_df["Functional Max"] = pd.to_numeric(expanded_df["Functional Max"], errors="coerce").fillna(0).astype(np.float64)

    # Ensure that Functional Max is not missing before upload
    if "Functional Max" not in expanded_df.columns:
        logging.error("❌ ERROR: 'Functional Max' column is missing from expanded_df after assignment.")

    expanded_df["Functional Max"] = expanded_df["Functional Max"].astype(np.float64)

    # Add Strength Change column
    expanded_df["Strength Change"] = expanded_df.apply(
        lambda row: +1 if (row["Functional Max"] - row["Tested Max"]) > 1 else
                    -1 if (row["Functional Max"] - row["Tested Max"]) < -1 else
                    0,
        axis=1
    )

    # Add Expected Change column using the evaluation function
    expanded_df["Expected Change"] = expanded_df.apply(evaluate_functional_max_adjustment, axis=1)

    # Add Mismatch column to identify discrepancies
    expanded_df["Mismatch"] = expanded_df["Strength Change"] != expanded_df["Expected Change"]

    return expanded_df

def exp_decay(reps, a, b, c):
    return a * np.exp(-b * reps) + c

def evaluate_functional_max_adjustment(row):
    """
    Evaluates whether Functional Max should increase, decrease, or stay the same based on 
    the per-exercise empirical reps vs weight curve.
    """
    try:
        tested_max = float(row["Tested Max"])
        assigned_reps = float(row["# of Reps"])
        assigned_weight = float(row["Assigned Weight"])
        simulated_reps = float(row["Simulated Reps"])
        simulated_weight = float(row["Simulated Weight"])
        exercise_code = row["Code"]
    except Exception as e:
        logging.error(f"❌ Error converting row values for evaluation: {e}")
        return 0

    if tested_max == 0 or assigned_reps == 0 or assigned_weight == 0:
        # logging.warning("⚠️ Insufficient data to evaluate functional max adjustment.")
        return 0

    coeffs = multiplier_fits.get(exercise_code)
    if coeffs is None:
        # logging.warning(f"⚠️ No coefficients found for exercise code {exercise_code}. Defaulting to 0 adjustment.")
        return 0

    a_fit, b_fit, c_fit, _ = coeffs[:4]

    expected_relative_weight_sim = exp_decay(simulated_reps, a_fit, b_fit, c_fit)
    assigned_relative_weight = assigned_weight / tested_max
    simulated_relative_weight = simulated_weight / tested_max

    if simulated_relative_weight > assigned_relative_weight:
        return +1
    elif simulated_relative_weight < assigned_relative_weight:
        return -1
    else:
        return 0

def evaluate_with_margin(row, margin=0.005, a_fit=0.85, b_fit=0.07, c_fit=0.1):
    """
    Evaluate Functional Max adjustment with a tolerance margin.
    """
    try:
        tested_max = float(row["Tested Max"])
        assigned_weight = float(row["Assigned Weight"])
        simulated_weight = float(row["Simulated Weight"])
        simulated_reps = float(row["Simulated Reps"])
    except Exception as e:
        logging.error(f"❌ Error converting row values for margin evaluation: {e}")
        return 0

    if tested_max == 0 or assigned_weight == 0:
        # logging.warning("⚠️ Insufficient data for margin evaluation.")
        return 0

    assigned_relative_weight = assigned_weight / tested_max
    simulated_relative_weight = simulated_weight / tested_max

    diff = simulated_relative_weight - assigned_relative_weight

    if diff > margin:
        return +1
    elif diff < -margin:
        return -1
    else:
        return 0
