import sys
import os
import pickle
import pandas as pd
import numpy as np  # Ensure NumPy is imported
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Dynamically add the NTlifting root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)
# logging.info(f"✅ Root directory added to sys.path: {ROOT_DIR}")

# Import necessary functions
from execution.helpers.data_loading import load_data
from execution.helpers.data_processing import preprocess_data

# Load raw data from Google Sheets
program_df, core_maxes_df = load_data()

# Run preprocessing step
flattened_core_maxes_df, repeated_program_df = preprocess_data(program_df, core_maxes_df)
logging.info("✅ Data successfully preprocessed!")

# Load fitted multipliers from pickle file
pickle_path = os.path.join(ROOT_DIR, "execution/helpers/multiplier_fits.pkl")
with open(pickle_path, "rb") as f:
    multiplier_fits = pickle.load(f)
logging.info("✅ multiplier_fits loaded successfully!")

# Merge repeated_program_df with flattened_core_maxes_df
merged_df = repeated_program_df.merge(flattened_core_maxes_df, on=["Player", "Relevant Core"], how="left")
logging.info("✅ Merged repeated_program_df with flattened_core_maxes_df")

# Apply multipliers to generate M_assigned efficiently
for exercise, params in multiplier_fits.items():
    mask = merged_df["Exercise"] == exercise
    if isinstance(params, float):  # Static multiplier
        merged_df.loc[mask, "M_assigned"] = params
    else:  # Dynamic multiplier
        w, s, r = merged_df.loc[mask, ["Week #", "Set #", "# of Reps"]].values.T
        merged_df.loc[mask, "M_assigned"] = params[0] * w + params[1] * s + params[2] * np.log(r + 1) + params[3]

# Ensure 'NRM' values remain labeled correctly
if "Tested Max" in merged_df.columns:
    merged_df["M_assigned"] = merged_df["M_assigned"].astype(object)
# else:
    # logging.error("❌ ERROR: 'Tested Max' column not found in merged_df!")

# Validate assigned multipliers
validation_errors = [
    (exercise, np.max(np.abs(group["M_assigned"] - group["Multiplier of Max"])))
    for exercise, group in merged_df.groupby("Exercise") if group["M_assigned"].dtype != object
    and np.any(np.abs(group["M_assigned"] - group["Multiplier of Max"]) > 0.05)
]

# Save merged DataFrame
os.makedirs("data", exist_ok=True)
pkl_path = "data/repeated_program.pkl"
merged_df.to_pickle(pkl_path)
# logging.info(f"✅ Saved repeated program to {pkl_path}")

# Log validation errors if any exist
if validation_errors:
    df_validation_errors = pd.DataFrame(validation_errors, columns=["Exercise", "Max Error"])
    logging.warning("⚠️ Validation Errors (Above ±0.05):")
    logging.warning(df_validation_errors.to_string(index=False))

def merge_data(repeated_program_df, flattened_core_maxes_df, multiplier_fits):
    """ Merges expanded program data with core maxes and applies multipliers. """

    logging.info("🔄 Running merge_data() function...")

    # Debugging: Log before merge
    # logging.info(f"🔍 Columns in repeated_program_df BEFORE merging: {list(repeated_program_df.columns)}")
    # logging.info(f"🔍 Columns in flattened_core_maxes_df BEFORE merging: {list(flattened_core_maxes_df.columns)}")

    # Perform the merge
    merged_df = repeated_program_df.merge(flattened_core_maxes_df, on=["Player", "Relevant Core"], how="left")

    # Fix: Check for duplicate `Tested Max` columns and rename properly
    if "Tested Max_x" in merged_df.columns and "Tested Max_y" in merged_df.columns:
        # logging.warning("⚠️ Detected duplicate `Tested Max` columns! Resolving issue...")

        # Use the `Tested Max_y` column and drop the other
        merged_df["Tested Max"] = merged_df["Tested Max_y"].fillna(merged_df["Tested Max_x"])
        merged_df.drop(columns=["Tested Max_x", "Tested Max_y"], inplace=True)

    # Debugging: Ensure `Tested Max` is present after merge
    # if "Tested Max" not in merged_df.columns:
        # logging.error("❌ ERROR: 'Tested Max' column missing after merge!")

    logging.info(f"✅ Merged successfully, 'Tested Max' available in merged_data.")

    return merged_df
