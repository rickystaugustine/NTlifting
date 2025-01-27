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
logging.info(f"✅ Root directory added to sys.path: {ROOT_DIR}")

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
else:
    logging.error("❌ ERROR: 'Tested Max' column not found in merged_df!")

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
logging.info(f"✅ Saved repeated program to {pkl_path}")

# Log validation errors if any exist
if validation_errors:
    df_validation_errors = pd.DataFrame(validation_errors, columns=["Exercise", "Max Error"])
    logging.warning("⚠️ Validation Errors (Above ±0.05):")
    logging.warning(df_validation_errors.to_string(index=False))

def merge_data():
    """ Merges repeated program data with core maxes and applies multipliers. """
    global merged_df  # Ensure function modifies the global dataframe
    
    logging.info("🔄 Running merge_data() function...")

    # Merge datasets
    merged_df = repeated_program_df.merge(flattened_core_maxes_df, on=["Player", "Relevant Core"], how="left")
    logging.info("✅ Merged repeated_program_df with flattened_core_maxes_df")

    # Apply multipliers to generate assigned weights
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
    else:
        logging.error("❌ ERROR: 'Tested Max' column not found in merged_df!")

    return merged_df.to_dict(orient="records")  # Convert DataFrame to a list of dictionaries
