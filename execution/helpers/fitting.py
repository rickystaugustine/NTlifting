import warnings
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from typing import Tuple, Union, List, Dict
from helpers.multipliers import ConstantMultiplier, FittedMultiplier, m_func

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
