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

    # Ensure 'Code' column exists to avoid KeyError
    if "Code" not in program_df.columns:
        logging.error("❌ ERROR: 'Code' column is missing in program_df! Adding a default column...")
        program_df["Code"] = "Unknown"  # Add a default value to prevent failure

    exercises = program_df["Code"].unique()
    exercise_functions = {}

    logging.info(f"🔍 Fitting multipliers for {len(exercises)} exercises...")

    for exercise in exercises:
        exercise_df = program_df[program_df["Code"] == exercise]

        if exercise_df.empty:
            logging.warning(f"⚠️ No data available for exercise: {exercise}")
            continue

        # Extract necessary columns, ensuring they exist
        required_columns = ["Week #", "Set #", "# of Reps", "Tested Max"]
        for col in required_columns:
            if col not in exercise_df.columns:
                logging.error(f"❌ ERROR: Missing required column '{col}' in program_df!")
                return None  # Fail gracefully

        w = exercise_df["Week #"].values
        s = exercise_df["Set #"].values
        r = exercise_df["# of Reps"].values
        maxes = exercise_df["Tested Max"].values

        # Handle potential missing or non-numeric values
        valid_mask = ~np.isnan(w) & ~np.isnan(s) & ~np.isnan(r) & ~np.isnan(maxes)
        if not valid_mask.any():
            logging.warning(f"⚠️ No valid numeric data for exercise: {exercise}")
            continue

        # Fit a simple linear model (placeholding as an example)
        coeffs = np.polyfit(w[valid_mask] + s[valid_mask] + np.log(r[valid_mask] + 1), maxes[valid_mask], 1)
        
        # Store the function for this exercise
        exercise_functions[exercise] = lambda w, s, r: coeffs[0] * (w + s + np.log(r + 1)) + coeffs[1]

        logging.info(f"✅ Fitted multipliers for {exercise}: {coeffs}")

    return exercise_functions