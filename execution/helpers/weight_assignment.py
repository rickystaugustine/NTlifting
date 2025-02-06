import pandas as pd
import numpy as np
import logging
import sys
import os

# Explicitly add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from execution.helpers.multipliers import multiplier_fits, exercise_functions
from execution.helpers.google_sheets_utils import write_to_google_sheet

logging.basicConfig(level=logging.INFO)

# Define paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
repeated_program_path = os.path.join(ROOT_DIR, "data/repeated_program.pkl")
core_maxes_path = os.path.join(ROOT_DIR, "data/flattened_core_maxes.pkl")

def assign_weights(merged_data, flattened_core_maxes_df, exercise_functions):
    """ Assigns weights based on merged data, core maxes, and exercise multipliers. """

    logging.info("🚀 Assigning weights using merged data and multipliers...")

    if merged_data.empty:
        logging.error("❌ ERROR: merged_data is empty. Check data preprocessing.")
        return pd.DataFrame()

    assigned_df = merged_data.copy()

    # Ensure "Assigned Weight" column exists
    assigned_df["Assigned Weight"] = np.nan

    # Convert "Tested Max" and "Multiplier of Max" to numeric
    assigned_df["Tested Max"] = pd.to_numeric(assigned_df["Tested Max"], errors="coerce")
    assigned_df["Multiplier of Max"] = pd.to_numeric(assigned_df["Multiplier of Max"], errors="coerce")

    # Debugging: Check if `exercise_functions` is loaded properly
    if not exercise_functions:
        logging.error("❌ ERROR: `exercise_functions` is EMPTY! Multipliers were not fitted.")
        return assigned_df  # Return unchanged data to avoid breaking pipeline

    # Log valid vs. invalid rows
    num_valid = assigned_df["Tested Max"].notna().sum()
    num_invalid = assigned_df["Tested Max"].isna().sum()
    logging.info(f"✅ Valid Tested Max entries: {num_valid}")
    logging.warning(f"⚠️ Invalid Tested Max entries (will be 'NRM'): {num_invalid}")

    for exercise, function in exercise_functions.items():
        mask = assigned_df["Exercise"] == exercise

        # Ensure "Assigned Weight" column is object type before assigning strings
        assigned_df["Assigned Weight"] = assigned_df["Assigned Weight"].astype(object)

        # Assign "NRM" to missing values
        assigned_df.loc[mask & (assigned_df["Tested Max"].isna()), "Assigned Weight"] = "NRM"

        # Debugging: Check number of valid weights before calculation
        valid_mask = mask & (~assigned_df["Tested Max"].isna()) & (~assigned_df["Multiplier of Max"].isna())
        valid_count = valid_mask.sum()
        # logging.info(f"✅ Valid weights for {exercise}: {valid_count}")

        if function is not None and valid_count > 0:
            assigned_df.loc[valid_mask, "Assigned Weight"] = (
                assigned_df.loc[valid_mask, "Tested Max"] * assigned_df.loc[valid_mask, "Multiplier of Max"]
            )

    # Replace NaN values with "NRM"
    assigned_df["Assigned Weight"] = assigned_df["Assigned Weight"].astype(object)
    assigned_df.loc[assigned_df["Tested Max"].isna(), "Assigned Weight"] = "NRM"

    logging.info(f"✅ Assigned weights successfully calculated. Non-NRM count: {(assigned_df['Assigned Weight'] != 'NRM').sum()}")

    return assigned_df

# ✅ Ensure script doesn't auto-execute when imported
if __name__ == "__main__":
    assign_weights()
