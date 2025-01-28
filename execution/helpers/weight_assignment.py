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

def assign_weights():
    """Calculates and assigns weights based on multipliers and maxes."""
    logging.info("Loading data for weight assignment...")
    expected_files = [repeated_program_path, core_maxes_path]
    missing_files = [f for f in expected_files if not os.path.exists(f)]

    if missing_files:
        raise FileNotFoundError(f"❌ ERROR: Missing required data files: {missing_files}")

    repeated_program_df = pd.read_pickle(repeated_program_path)
    flattened_core_maxes_df = pd.read_pickle(core_maxes_path)

    logging.info("Merging repeated program with core maxes...")
    merged_df = repeated_program_df.merge(flattened_core_maxes_df, on=["Player", "Relevant Core"], how="left")

    # Ensure "Tested Max" column is present
    if "Tested Max" not in merged_df.columns:
        raise KeyError("❌ ERROR: 'Tested Max' column is missing from merged_df after merging!")

    # Compute Fitted Multipliers
    weeks, sets, reps, codes = (
        merged_df["Week #"].values,
        merged_df["Set #"].values,
        merged_df["# of Reps"].values,
        merged_df["Code"].values,
    )

    multipliers = [
        exercise_functions.get(int(code), lambda w, s, r: 0)(w, s, r)
        for code, w, s, r in zip(codes, weeks, sets, reps)
    ]
    merged_df["Fitted Multiplier"] = multipliers

    # Calculate Assigned Weights
    merged_df["Assigned Weight"] = np.floor(pd.to_numeric(merged_df["Tested Max"], errors="coerce") * merged_df["Fitted Multiplier"] / 5) * 5
    merged_df["Assigned Weight"] = merged_df["Assigned Weight"].astype(object)

    # Handle NRM Cases
    merged_df.loc[merged_df["Tested Max"] == "NRM", "Assigned Weight"] = "NRM"

    # Save assigned weights
    assigned_weights_path = os.path.join(ROOT_DIR, "data/assigned_weights.pkl")
    merged_df.to_pickle(assigned_weights_path)
    logging.info(f"✅ Assigned weights saved to {assigned_weights_path}")

    # Upload to Google Sheets
    logging.info("Uploading assigned weights to Google Sheets...")
    write_to_google_sheet("After-School Lifting", "AssignedWeights", merged_df)
    logging.info("✅ Assigned weights successfully saved to Google Sheets!")

    return merged_df

# ✅ Ensure script doesn't auto-execute when imported
if __name__ == "__main__":
    assign_weights()
