import os
import pickle
import logging
import numpy as np
import time
import sys
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

start_time = time.time()
logging.info(f"Beginning helper function import...")

# Add the execution directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helpers.data_loading import load_data
from helpers.data_processing import preprocess_data
from helpers.multiplier_fitting import fit_multipliers
from helpers.exercise_fitting import fit_exercise_multipliers
from helpers.data_merging import merge_data
from helpers.weight_assignment import assign_weights
from helpers.simulation import run_simulation
from helpers.google_sheets_utils import write_to_google_sheet  # Ensure this function exists

step_time = time.time() - start_time
logging.info(f"✅ Helper-function import completed in {step_time:.2f} seconds.")

def upload_dataframe(df, sheet_name):
    try:
        write_to_google_sheet("After-School Lifting", sheet_name, df)
        logging.info(f"✅ Successfully uploaded {sheet_name} to Google Sheets.")
    except Exception as e:
        logging.error(f"❌ ERROR: Failed to upload {sheet_name} to Google Sheets - {str(e)}")

if __name__ == "__main__":
    # Step 1: Load Data
    logging.info(f"Loading data...")
    step_start_time = time.time()
    program_df, maxes_df = load_data()
    step_time = time.time() - step_start_time
    logging.info(f"✅ Data loading completed in {step_time:.2f} seconds.")

    # Step 2: Preprocess Data
    logging.info(f"Preprocessing data...")
    step_start_time = time.time()
    flattened_core_maxes_df, repeated_program_df = preprocess_data(program_df, maxes_df)
    step_time = time.time() - step_start_time
    logging.info(f"✅ Data preprocessing completed in {step_time:.2f} seconds.")

    # Step 3: Fit Multipliers
    logging.info(f"Fitting multipliers...")
    step_start_time = time.time()
    exercise_functions = fit_multipliers(repeated_program_df)  # Now only returns multipliers
    _, constant_multiplier_exercises = fit_exercise_multipliers(repeated_program_df)  # Get classification separately
    constant_multiplier_exercises = {str(key).strip().upper(): value for key, value in constant_multiplier_exercises.items()}
    step_time = time.time() - step_start_time
    logging.info(f"✅ Multiplier fitting completed in {step_time:.2f} seconds.")

    # Step 4: Merge Data
    logging.info(f"Merging data...")
    step_start_time = time.time()
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    pickle_path = os.path.join(ROOT_DIR, "helpers/multiplier_fits.pkl")

    try:
        with open(pickle_path, "rb") as f:
            multiplier_fits = pickle.load(f)
    except FileNotFoundError:
        logging.error("❌ ERROR: multiplier_fits.pkl not found! Ensure multipliers were fitted and saved.")
        multiplier_fits = {}

    merged_data = merge_data(repeated_program_df, flattened_core_maxes_df, multiplier_fits)

    # ✅ Apply memory optimizations before analyzing usage
    merged_data["Exercise"] = merged_data["Exercise"].astype("category")
    merged_data["Relevant Core"] = merged_data["Relevant Core"].astype("category")
    merged_data["Player"] = merged_data["Player"].astype("category")
    merged_data["Tested Max"] = merged_data["Tested Max"].astype("category")  # If categorical, otherwise convert to numeric

    merged_data["Code"] = pd.to_numeric(merged_data["Code"], downcast="integer")
    merged_data["Week #"] = pd.to_numeric(merged_data["Week #"], downcast="integer")
    merged_data["Set #"] = pd.to_numeric(merged_data["Set #"], downcast="integer")
    merged_data["# of Reps"] = pd.to_numeric(merged_data["# of Reps"], downcast="integer")

    merged_data["Multiplier of Max"] = pd.to_numeric(merged_data["Multiplier of Max"], downcast="float")

    step_time = time.time() - step_start_time
    logging.info(f"✅ Data merging completed in {step_time:.2f} seconds.")

    # Step 5: Calculate Assigned Weights
    logging.info(f"Assigning weights...")
    step_start_time = time.time()
    assigned_weights_df = assign_weights(merged_data, flattened_core_maxes_df, exercise_functions)
    # Ensure categorical columns allow "NRM" as a category before filling NaN values
    for col in assigned_weights_df.select_dtypes(include="category").columns:
        assigned_weights_df[col] = assigned_weights_df[col].cat.add_categories("NRM")

    # Now safely replace NaN values with "NRM"
    assigned_weights_df = assigned_weights_df.fillna("NRM").astype(str)

    if "Tested Max" not in assigned_weights_df.columns:
        logging.error("❌ ERROR: 'Tested Max' missing in assigned_weights_df before simulation!")

    # Replace NaN values with "NRM" before uploading
    assigned_weights_df["Assigned Weight"] = assigned_weights_df["Assigned Weight"].astype(object)
    assigned_weights_df = assigned_weights_df.fillna("NRM").astype(str)

    # Upload to Google Sheets
    logging.info(f"Uploading Assigned Weights to Google Sheets...")
    upload_dataframe(assigned_weights_df, "AssignedWeights")

    step_time = time.time() - step_start_time
    logging.info(f"✅ Weight assignment completed in {step_time:.2f} seconds.")

    # Step 6: Simulate Lifting Performance
    logging.info(f"Simulating lift data...")
    step_start_time = time.time()
    simulated_data = run_simulation(assigned_weights_df, maxes_df, constant_multiplier_exercises)

    step_time = time.time() - step_start_time
    logging.info(f"✅ Lift-data simulation completed in {step_time:.2f} seconds.")

    logging.info(f"Uploading Simulated Data to Google Sheets...")
    upload_dataframe(simulated_data, "SimulatedData")

    run_time = time.time() - start_time
    logging.info(f"🏁 Execution completed successfully in {run_time:.2f} seconds.")
