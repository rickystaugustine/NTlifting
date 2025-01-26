import pandas as pd
import numpy as np
import logging
import sys
import os

# Explicitly add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("✅ Root directory added to sys.path:", project_root)
print("🔍 sys.path:", sys.path)

from execution.helpers.multipliers import multiplier_fits, exercise_functions
from execution.helpers.google_sheets_utils import write_to_google_sheet

logging.basicConfig(level=logging.INFO)

# Define paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
repeated_program_path = os.path.join(ROOT_DIR, "data/repeated_program.pkl")
core_maxes_path = os.path.join(ROOT_DIR, "data/flattened_core_maxes.pkl")

# Load Data
logging.info("Loading data for weight assignment...")
expected_files = [repeated_program_path, core_maxes_path]
missing_files = [f for f in expected_files if not os.path.exists(f)]

if missing_files:
    print(f"❌ ERROR: The following required files are missing:\n {missing_files}")
    raise FileNotFoundError("❌ ERROR: One or more required data files are missing!")

repeated_program_df = pd.read_pickle(repeated_program_path)
flattened_core_maxes_df = pd.read_pickle(core_maxes_path)

# Debugging: Check if "Tested Max" exists before merging
print("✅ DEBUG: Checking if 'Tested Max' exists in flattened_core_maxes_df")
print(flattened_core_maxes_df.columns)

# Merge Data
logging.info("Merging repeated program with core maxes...")
merged_df = repeated_program_df.merge(flattened_core_maxes_df, on=["Player", "Relevant Core"], how="left")

# 🛠 **Fix: Ensure "Tested Max" is properly handled**
if "Tested Max_y" in merged_df.columns:
    merged_df.rename(columns={"Tested Max_y": "Tested Max"}, inplace=True)
    merged_df.drop(columns=["Tested Max_x"], errors="ignore", inplace=True)

# Ensure "Tested Max" column is present
if "Tested Max" not in merged_df.columns:
    raise KeyError("❌ ERROR: 'Tested Max' column is missing from merged_df after merging!")

# 🛠 **Fix: Ensure "M_assigned" is correctly renamed**
if "M_assigned" in merged_df.columns:
    merged_df.rename(columns={"M_assigned": "Fitted Multiplier"}, inplace=True)

# Debugging: Check if "Tested Max" exists after cleaning
print("✅ DEBUG: Checking columns in merged_df after renaming")
print(merged_df.columns)

# Calculate Multipliers & Assigned Weights
logging.info("Calculating multipliers and assigned weights...")
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

# Convert Assigned Weight to object before setting NRM
merged_df["Assigned Weight"] = np.floor(pd.to_numeric(merged_df["Tested Max"], errors="coerce") * merged_df["Fitted Multiplier"] / 5) * 5
merged_df["Assigned Weight"] = merged_df["Assigned Weight"].astype(object)  # Ensures mixed types are allowed

# Handle NRM Cases
merged_df.loc[merged_df["Tested Max"] == "NRM", "Assigned Weight"] = "NRM"

# Save assigned weights to a file
assigned_weights_path = os.path.join(ROOT_DIR, "data/assigned_weights.pkl")
merged_df.to_pickle(assigned_weights_path)
logging.info(f"✅ Assigned weights saved to {assigned_weights_path}")

# Save & Upload Data
logging.info("Uploading assigned weights to Google Sheets...")
write_to_google_sheet(sheet_name="After-School Lifting", worksheet_name="AssignedWeights", data=merged_df)
logging.info("✅ Assigned weights successfully saved to Google Sheets!")
