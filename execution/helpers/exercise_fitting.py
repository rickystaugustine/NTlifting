import logging
import warnings
import pandas as pd
import numpy as np

import sys
import os
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Dynamically add the NTlifting root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)

from scipy.optimize import curve_fit, OptimizeWarning
from typing import Tuple, Union, List, Dict
from execution.helpers.multipliers import ConstantMultiplier, FittedMultiplier, m_func
from concurrent.futures import ProcessPoolExecutor, as_completed

def fit_single_exercise_global(code: int, program_records: List[Dict]) -> Tuple[int, Union[ConstantMultiplier, FittedMultiplier]]:
    """
    Fit a multiplier function for a single exercise.
    Returns:
        (exercise_code, multiplier_instance)
    """
    program_df = pd.DataFrame(program_records)
    exercise_df = program_df[program_df['Code'] == code]
    weeks, sets, reps = exercise_df['Week #'].values, exercise_df['Set #'].values, exercise_df['# of Reps'].values
    multipliers = exercise_df['Multiplier of Max'].values

    if len(set(multipliers)) == 1:
        return code, ConstantMultiplier(multipliers[0])

    if len(weeks) == 0 or len(sets) == 0 or len(reps) == 0:
        logging.error(f"❌ ERROR: Missing data for {code}. Cannot perform curve fitting.")
        return code, ConstantMultiplier(multipliers.mean())

    initial_guess = np.polyfit(weeks + sets + np.log(reps + 1), multipliers, 1).tolist()
    while len(initial_guess) < 4:
        initial_guess.append(multipliers.mean())  # Ensure p0 has at least 4 elements
    bounds = ([0] * len(initial_guess), [np.inf] * len(initial_guess))

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=OptimizeWarning)
            popt, _ = curve_fit(m_func, (weeks, sets, reps), multipliers, p0=initial_guess, bounds=bounds, maxfev=10000)
        return code, FittedMultiplier(popt)
    except (OptimizeWarning, RuntimeError) as e:
        logging.error(f"❌ ERROR: Curve fitting failed for exercise {code}: {e}. Assigning mean multiplier.")
        return code, ConstantMultiplier(multipliers.mean())

def fit_exercise_multipliers(program_df):
    """Fits multipliers for each exercise, classifies them as constant (2a) or dynamic (2b), 
       and stores a backup linear model for additional calculations."""

    # Ensure program_df is a Pandas DataFrame
    if not isinstance(program_df, pd.DataFrame):
        program_df = pd.DataFrame(program_df)

    # Ensure required columns exist
    required_columns = ["Code", "Week #", "Set #", "# of Reps", "Multiplier of Max", "Tested Max"]
    missing_columns = [col for col in required_columns if col not in program_df.columns]

    if missing_columns:
        logging.error(f"❌ ERROR: Missing required columns in program_df: {missing_columns}")
        return {}, {}  # Return empty dictionaries

    exercises = program_df["Code"].unique()
    exercise_functions = {}  # Stores multiplier functions

    # logging.info(f"🔍 Fitting multipliers and classifying {len(exercises)} exercises...")

    for exercise in exercises:
        exercise_df = program_df[program_df["Code"] == exercise]

        if exercise_df.empty:
            logging.error(f"❌ ERROR: No valid data available for exercise: {exercise}")
            continue

        # Convert values to float to prevent TypeError
        w = pd.to_numeric(exercise_df["Week #"], errors="coerce").values
        s = pd.to_numeric(exercise_df["Set #"], errors="coerce").values
        r = pd.to_numeric(exercise_df["# of Reps"], errors="coerce").values
        maxes = pd.to_numeric(exercise_df["Tested Max"], errors="coerce").values

        # Call fit_single_exercise_global() to determine multiplier type
        code, multiplier_instance = fit_single_exercise_global(exercise, program_df.to_dict("records"))

        # Store the function for this exercise
        if isinstance(multiplier_instance, FittedMultiplier):
            exercise_functions[exercise] = multiplier_instance  # Store full object
        elif isinstance(multiplier_instance, ConstantMultiplier):
            exercise_functions[exercise] = multiplier_instance  # Store full object


    # logging.info(f"DEBUG: First 10 keys in exercise_functions: {list(exercise_functions.keys())[:10]}")
    # for exercise, function in exercise_functions.items():
        # logging.info(f"DEBUG: {exercise} → Stored Function: {repr(function)}")

    # logging.info(f"✅ Fitted multipliers successfully created for {len(exercise_functions)} exercises.")
    # logging.info(f"✅ Exercises successfully classified as constant (2a) or fitted (2b).")

    exercise_functions = {
        str(program_df.loc[program_df["Code"] == exercise, "Exercise"].iloc[0]).strip().upper(): value
        for exercise, value in exercise_functions.items()
    }

    return exercise_functions
