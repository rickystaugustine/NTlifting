import os
import pickle
import logging
import numpy as np
from helpers.data_loading import load_data
from helpers.data_processing import preprocess_data
from helpers.multiplier_fitting import fit_multipliers
from helpers.data_merging import merge_data
from helpers.weight_assignment import assign_weights
from helpers.simulation import run_simulation
from execution.helpers.google_sheets_utils import write_to_google_sheet

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    logging.info("🚀 Starting Lifting Program Execution...")

    # Step 1: Load Data
    program_df, maxes_df = load_data()

    # Step 2: Preprocess Data
    logging.info(f"🔍 Column names in CompleteProgram (program_df): {list(program_df.columns)}")
    logging.info(f"🔍 Column names in Maxes (maxes_df): {list(maxes_df.columns)}")
    flattened_core_maxes_df, repeated_program_df = preprocess_data(program_df, maxes_df)

    # Step 3: Fit Multipliers
    exercise_functions = fit_multipliers(repeated_program_df)

    # Step 4: Merge Data
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    pickle_path = os.path.join(ROOT_DIR, "helpers/multiplier_fits.pkl")

    try:
        with open(pickle_path, "rb") as f:
            multiplier_fits = pickle.load(f)
        logging.info("✅ multiplier_fits loaded successfully!")
    except FileNotFoundError:
        logging.error("❌ ERROR: multiplier_fits.pkl not found! Ensure multipliers were fitted and saved.")
        multiplier_fits = {}

    # Merge processed program data
    logging.info(f"🔍 Columns in repeated_program_df BEFORE merge_data: {list(repeated_program_df.columns)}")
    logging.info(f"🔍 Columns in flattened_core_maxes_df BEFORE merge_data: {list(flattened_core_maxes_df.columns)}")

    # Debugging: Sample data before merging
    logging.info(f"🔍 Sample repeated_program_df: \n{repeated_program_df.head()}")
    logging.info(f"🔍 Sample flattened_core_maxes_df: \n{flattened_core_maxes_df.head()}")

    merged_data = merge_data(repeated_program_df, flattened_core_maxes_df, multiplier_fits)
    logging.info(f"🔍 DEBUG: Columns in merged_data before assign_weights: {list(merged_data.columns)}")

    # Step 5: Calculate Assigned Weights
    assigned_weights_df = assign_weights(merged_data, flattened_core_maxes_df, exercise_functions)

    # Debugging: Check for NaN values before upload
    if assigned_weights_df.isna().sum().sum() > 0:
        logging.warning(f"⚠️ WARNING: NaN values present before upload: \n{assigned_weights_df.isna().sum()}")
        logging.warning(f"🔍 Sample NaN rows: \n{assigned_weights_df[assigned_weights_df.isna().any(axis=1)]}")

    # Replace NaN values with "NRM" before uploading
    assigned_weights_df["Assigned Weight"] = assigned_weights_df["Assigned Weight"].astype(object)
    assigned_weights_df = assigned_weights_df.fillna("NRM").astype(str)

    # Verify after replacement
    if assigned_weights_df.isna().sum().sum() == 0:
        logging.info("✅ All NaN values successfully replaced before upload.")

    # Upload to Google Sheets
    write_to_google_sheet("After-School Lifting", "AssignedWeights", assigned_weights_df)
    logging.info("✅ Assigned weights successfully uploaded to Google Sheets!")

    # Step 6: Simulate Lifting Performance
    simulated_data = run_simulation(assigned_weights_df)

    # Convert all NumPy int64 values to Python int before uploading
    simulated_data = simulated_data.map(lambda x: int(x) if isinstance(x, (np.int64, np.int32)) else x)

    # Write Simulated Data to Google Sheets
    write_to_google_sheet("After-School Lifting", "SimulatedData", simulated_data)
    logging.info("✅ Simulated data successfully uploaded to Google Sheets!")

    logging.info("🏁 Execution completed successfully!")
