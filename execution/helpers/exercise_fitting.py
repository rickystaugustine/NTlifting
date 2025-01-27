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
logging.info(f"✅ Root directory added to sys.path: {ROOT_DIR}")

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
        logging.info(f"Exercise {code} has a constant multiplier: {multipliers[0]}")
        return code, ConstantMultiplier(multipliers[0])

    initial_guess = np.polyfit(weeks + sets + np.log(reps + 1), multipliers, 1).tolist() + [multipliers.mean()]
    bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=OptimizeWarning)
            popt, _ = curve_fit(m_func, (weeks, sets, reps), multipliers, p0=initial_guess, bounds=bounds, maxfev=10000)
        logging.info(f"Exercise {code} fitted successfully with coefficients: {popt}")
        return code, FittedMultiplier(popt)
    except (OptimizeWarning, RuntimeError) as e:
        logging.warning(f"Curve fitting failed for exercise {code}: {e}. Assigning mean multiplier.")
        return code, ConstantMultiplier(multipliers.mean())

def fit_exercise_multipliers(program_df):
    """ Fits multipliers for each exercise. """

    # Ensure program_df is a Pandas DataFrame
    if not isinstance(program_df, pd.DataFrame):
        logging.warning("⚠️ program_df was passed as a dict, converting to DataFrame...")
        program_df = pd.DataFrame(program_df)

    # Ensure required columns exist
    required_columns = ["Code", "Week #", "Set #", "# of Reps", "Tested Max"]
    missing_columns = [col for col in required_columns if col not in program_df.columns]
    
    if missing_columns:
        logging.error(f"❌ ERROR: Missing required columns in program_df: {missing_columns}")
        return {}  # Return an empty dictionary instead of None to prevent TypeError

    exercises = program_df["Code"].unique()
    exercise_functions = {}

    logging.info(f"🔍 Fitting multipliers for {len(exercises)} exercises...")

    for exercise in exercises:
        exercise_df = program_df[program_df["Code"] == exercise]

        if exercise_df.empty:
            logging.warning(f"⚠️ No data available for exercise: {exercise}")
            continue

        w = exercise_df["Week #"].values
        s = exercise_df["Set #"].values
        r = exercise_df["# of Reps"].values
        maxes = exercise_df["Tested Max"].values

        # Handle potential missing or non-numeric values
        valid_mask = ~pd.isna(w) & ~pd.isna(s) & ~pd.isna(r) & ~pd.isna(maxes)
        if not valid_mask.any():
            logging.warning(f"⚠️ No valid numeric data for exercise: {exercise}")
            continue

        # Fit a simple linear model
        coeffs = np.polyfit(w[valid_mask] + s[valid_mask] + np.log(r[valid_mask] + 1), maxes[valid_mask], 1)
        
        # Store the function for this exercise
        exercise_functions[exercise] = lambda w, s, r: coeffs[0] * (w + s + np.log(r + 1)) + coeffs[1]

        logging.info(f"✅ Fitted multipliers for {exercise}: {coeffs}")

    return exercise_functions  # Ensure this always returns a dictionary
