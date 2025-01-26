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
