import logging
import warnings
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from typing import Tuple, Union, List, Dict
from helpers.multipliers import ConstantMultiplier, FittedMultiplier, m_func
from concurrent.futures import ProcessPoolExecutor, as_completed

def fit_single_exercise_global(code: int, program_records: List[Dict]) -> Tuple[int, Union[ConstantMultiplier, FittedMultiplier]]:
    """
    Fit a multiplier function for a single exercise.

    Returns:
        (exercise_code, multiplier_instance)
    """
    program_df = pd.DataFrame(program_records)
    exercise_df = program_df[program_df['Code'] == code]
    weeks, sets, reps = exercise_df['Week #'], exercise_df['Set #'], exercise_df['# of Reps']
    multipliers = exercise_df['Multiplier of Max']

    if len(set(multipliers)) == 1:
        return code, ConstantMultiplier(multipliers.iloc[0])

    initial_guess = [1, 1, 1, multipliers.mean()]
    bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=OptimizeWarning)
            popt, _ = curve_fit(m_func, (weeks, sets, reps), multipliers, p0=initial_guess, bounds=bounds, maxfev=10000)
        return code, FittedMultiplier(popt)
    except (OptimizeWarning, RuntimeError):
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
