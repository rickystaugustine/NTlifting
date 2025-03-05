import os
import pickle
import logging
import numpy as np
import time
import sys

start_func_import_time = time.time()
print(f"Beginning helper function import...")

# Add the execution directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helpers.data_loading import load_data
from helpers.data_processing import preprocess_data
from helpers.multiplier_fitting import fit_multipliers
from helpers.data_merging import merge_data
from helpers.weight_assignment import assign_weights
from helpers.simulation import run_simulation
from execution.helpers.google_sheets_utils import write_to_google_sheet
func_import_time = time.time() - start_func_import_time
print(f"Importing helper functions took {func_import_time:.2f} seconds")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    logging.info("🚀 Starting Lifting Program Execution...")
    start_time = time.time()

    # Step 1: Load Data
    program_df, maxes_df = load_data()
    load_time = time.time() - start_time
    print(f"Reading Sheets data took {load_time:.2f} seconds")

    # Step 2: Preprocess Data
    # logging.info(f"🔍 Column names in CompleteProgram (program_df): {list(program_df.columns)}")
    # logging.info(f"🔍 Column names in Maxes (maxes_df): {list(maxes_df.columns)}")
    start_preprocess_time = time.time()
    flattened_core_maxes_df, repeated_program_df = preprocess_data(program_df, maxes_df)
    preprocess_time = time.time() - start_preprocess_time
    print(f"Preprocessing data took {preprocess_time:.2f} seconds")

    # Step 3: Fit Multipliers
    start_fit_multipliers_time = time.time()
    exercise_functions = fit_multipliers(repeated_program_df)
    fit_multipliers_time = time.time() - start_fit_multipliers_time
    print(f"Fitting multipliers took {fit_multipliers_time:.2f} seconds")

    # Step 4: Merge Data
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    pickle_path = os.path.join(ROOT_DIR, "helpers/multiplier_fits.pkl")

    try:
        with open(pickle_path, "rb") as f:
            start_load_multiplier_fits_time = time.time()
            multiplier_fits = pickle.load(f)
            load_multiplier_fits_time = time.time() - start_load_multiplier_fits_time
        logging.info("✅ multiplier_fits loaded successfully...")
        print(f"...in {load_multiplier_fits_time:.2f} seconds")
    except FileNotFoundError:
        logging.error("❌ ERROR: multiplier_fits.pkl not found! Ensure multipliers were fitted and saved.")
        multiplier_fits = {}

    # Merge processed program data
    # logging.info(f"🔍 Columns in repeated_program_df BEFORE merge_data: {list(repeated_program_df.columns)}")
    # logging.info(f"🔍 Columns in flattened_core_maxes_df BEFORE merge_data: {list(flattened_core_maxes_df.columns)}")

    # Debugging: Sample data before merging
    # logging.info(f"🔍 Sample repeated_program_df: \n{repeated_program_df.head()}")
    # logging.info(f"🔍 Sample flattened_core_maxes_df: \n{flattened_core_maxes_df.head()}")

    start_merge_data_time = time.time()
    merged_data = merge_data(repeated_program_df, flattened_core_maxes_df, multiplier_fits)
    merge_data_time = time.time() - start_merge_data_time
    print(f"Merging data took {merge_data_time:.2f} seconds")
    # logging.info(f"🔍 DEBUG: Columns in merged_data before assign_weights: {list(merged_data.columns)}")

    # Step 5: Calculate Assigned Weights
    start_assign_weights_time = time.time()
    assigned_weights_df = assign_weights(merged_data, flattened_core_maxes_df, exercise_functions)
    assigned_weights_time = time.time() - start_assign_weights_time
    print(f"Assigning weights took {assigned_weights_time:.2f} seconds")

    # Debugging: Check for NaN values before upload
    # if assigned_weights_df.isna().sum().sum() > 0:
        # logging.warning(f"⚠️ WARNING: NaN values present before upload: \n{assigned_weights_df.isna().sum()}")
        # logging.warning(f"🔍 Sample NaN rows: \n{assigned_weights_df[assigned_weights_df.isna().any(axis=1)]}")

    # Replace NaN values with "NRM" before uploading
    start_NaN_replacement_time = time.time()
    assigned_weights_df["Assigned Weight"] = assigned_weights_df["Assigned Weight"].astype(object)
    assigned_weights_df = assigned_weights_df.fillna("NRM").astype(str)
    NaN_replacement_time = time.time() - start_NaN_replacement_time
    print(f"Replacing NaN values with NRM took {NaN_replacement_time:.2f} seconds")

    # Verify after replacement
    start_NaN_replacement_verification_time = time.time()
    if assigned_weights_df.isna().sum().sum() == 0:
        logging.info("✅ All NaN values successfully replaced before upload.")
    NaN_replacement_verification_time = time.time() - start_NaN_replacement_verification_time
    print(f"Verifying NaN replacement with NRM took {NaN_replacement_verification_time:.2f} seconds")

    # Upload to Google Sheets
    start_AssignedWeight_time = time.time()
    write_to_google_sheet("After-School Lifting", "AssignedWeights", assigned_weights_df)
    logging.info("✅ Assigned weights successfully uploaded to Google Sheets!")
    AssignedWeight_time = time.time() - start_AssignedWeight_time
    print(f"Writing to AssignedWeights took {AssignedWeight_time:.2f} seconds")

    # Step 6: Simulate Lifting Performance
    start_simulation_time = time.time()
    simulated_data = run_simulation(assigned_weights_df)
    simulation_time = time.time() - start_simulation_time
    print(f"Running simulation took {simulation_time:.2f} seconds")

    # Convert all NumPy int64 values to Python int before uploading
    start_int_conversion_time = time.time()
    simulated_data = simulated_data.map(lambda x: int(x) if isinstance(x, (np.int64, np.int32)) else x)
    int_conversion_time = time.time() - start_int_conversion_time
    print(f"Converting values to Python int took {int_conversion_time:.2f} seconds")

    # Write Simulated Data to Google Sheets
    start_SimulatedData_time = time.time()
    write_to_google_sheet("After-School Lifting", "SimulatedData", simulated_data)
    logging.info("✅ Simulated data successfully uploaded to Google Sheets!")
    SimulatedData_time = time.time() - start_SimulatedData_time
    print(f"Writing to SimulatedData took {SimulatedData_time:.2f} seconds")

    run_time = time.time() - start_time
    logging.info("🏁 Execution completed successfully...")
    print(f"...in {run_time:.2f} seconds")
