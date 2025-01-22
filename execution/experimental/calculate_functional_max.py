import sys
import os
import pandas as pd
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from helpers.google_sheets_utils import read_google_sheets, write_to_google_sheet
from helpers.fitting import fit_single_exercise_global
from helpers.simulation import simulate_iteration
from helpers.multipliers import ConstantMultiplier, FittedMultiplier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    logging.info("Reading data from Google Sheets...")
    program_df = read_google_sheets("After-School Lifting", "CompleteProgram")
    core_maxes_df = read_google_sheets("After-School Lifting", "Maxes")
    logging.info(f"Loaded {len(program_df)} program records and {len(core_maxes_df)} max records.")
    return program_df, core_maxes_df

def preprocess_data(program_df, core_maxes_df):
    logging.info("Preprocessing core maxes...")
    core_lifts = ["Bench", "Barbell Squat", "Clean", "Hex-Bar Deadlift"]
    flattened_core_maxes_df = core_maxes_df.melt(id_vars=["Player"], value_vars=core_lifts, var_name="Relevant Core", value_name="Tested Max")
    logging.info("Creating repeated program...")
    players = core_maxes_df["Player"].unique()
    repeated_program_df = pd.concat([program_df.assign(Player=player) for player in players], ignore_index=True)
    logging.info(f"Repeated program generated with {len(repeated_program_df)} records.")
    return flattened_core_maxes_df, repeated_program_df


def fit_exercise_multipliers(program_df):
    logging.info("Fitting functions by exercise...")
    exercises = program_df['Code'].unique()
    program_records = program_df.to_dict("records")
    exercise_functions = {}
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(fit_single_exercise_global, int(code), program_records): int(code) for code in exercises}
        for future in as_completed(futures):
            code = futures[future]
            try:
                func_tuple = future.result()
                exercise_functions[int(code)] = func_tuple
                logging.info(f"Fit completed for exercise code {code}.")
            except Exception as e:
                logging.error(f"Error processing exercise code {code}: {e}")
    logging.info("Exercise functions fitted successfully.")
    return exercise_functions

def calculate_assigned_weights(repeated_program_df, flattened_core_maxes_df, exercise_functions):
    logging.info("Calculating multipliers and assigned weights...")
    repeated_program_df = repeated_program_df.merge(flattened_core_maxes_df, on=["Player", "Relevant Core"], how="left")
    repeated_program_df["Tested Max"] = pd.to_numeric(repeated_program_df["Tested Max"], errors="coerce").fillna(0)
    weeks, sets, reps, codes = repeated_program_df["Week #"].values, repeated_program_df["Set #"].values, repeated_program_df["# of Reps"].values, repeated_program_df["Code"].values
    multipliers = [exercise_functions[int(code)][1](w, s, r) for code, w, s, r in zip(codes, weeks, sets, reps)]
    repeated_program_df["Multiplier"] = multipliers
    repeated_program_df["Assigned Weight"] = (np.floor(repeated_program_df["Tested Max"] * repeated_program_df["Multiplier"] / 5) * 5)
    logging.info(f"Assigned weights calculated for {len(repeated_program_df)} records.")
    return repeated_program_df

def simulate_lift_data(repeated_program_df, num_iterations=5):
    logging.info("Simulating lift data...")
    simulated_data = pd.concat([simulate_iteration(repeated_program_df) for _ in range(num_iterations)], ignore_index=True)
    logging.info(f"Simulated data generated with {len(simulated_data)} records.")
    return simulated_data

def calculate_functional_maxes(simulated_data):
    logging.info("Calculating functional maxes...")
    simulated_data["Functional Max"] = simulated_data["Actual Weight"] / simulated_data["Multiplier"]
    functional_maxes = simulated_data.groupby(["Player", "Relevant Core"])["Functional Max"].median().reset_index()
    logging.info("Functional maxes calculated successfully.")
    return functional_maxes

def main():
    logging.info("Starting process...")
    program_df, core_maxes_df = load_data()
    flattened_core_maxes_df, repeated_program_df = preprocess_data(program_df, core_maxes_df)
    exercise_functions = fit_exercise_multipliers(program_df)
    repeated_program_df = calculate_assigned_weights(repeated_program_df, flattened_core_maxes_df, exercise_functions)
    simulated_data = simulate_lift_data(repeated_program_df, num_iterations=5)
    functional_maxes = calculate_functional_maxes(simulated_data)
    logging.info("Process completed successfully!")
    write_to_google_sheet("After-School Lifting", "AssignedWeights", repeated_program_df)
    write_to_google_sheet("After-School Lifting", "SimulatedLiftData", simulated_data)
    write_to_google_sheet("After-School Lifting", "FunctionalMaxes", functional_maxes)
    logging.info("Results written to Google Sheets.")

if __name__ == "__main__":
    main()
