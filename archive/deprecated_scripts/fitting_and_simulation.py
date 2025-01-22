import warnings
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import norm
from typing import Tuple, Union, Dict, List

def m_func(inputs, a, b, c, d):
    """
    Multiplier function model.
    """
    w, s, r = inputs
    return a * w + b * s + c * np.log(r + 1) + d

# ─────────────────────────────────────────────────────────────────────────
# MULTIPLIER CLASSES
# ─────────────────────────────────────────────────────────────────────────
class ConstantMultiplier:
    """
    Represents a constant multiplier.
    """
    def __init__(self, value: float):
        self.value = value

    def __call__(self, w: float, s: float, r: float) -> float:
        return self.value


class FittedMultiplier:
    """
    Represents a fitted multiplier function.
    """
    def __init__(self, params: Tuple[float, float, float, float]):
        self.params = params

    def __call__(self, w: float, s: float, r: float) -> float:
        return m_func((w, s, r), *self.params)

# ─────────────────────────────────────────────────────────────────────────
# EXERCISE MODEL FITTING
# ─────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────
# REP DIFFERENCE SIMULATION
# ─────────────────────────────────────────────────────────────────────────
def generate_rep_differences_vectorized(r_assigned_array: np.ndarray) -> np.ndarray:
    """
    Generate rep differences based on a probability distribution.
    """
    size = len(r_assigned_array)
    min_bound, max_bound = -0.9 * r_assigned_array, 5.0 * r_assigned_array
    sigma_L, sigma_R = ((-1 * min_bound) * 0.997) / 3, (max_bound * 0.997) / 3

    x_vals = np.linspace(min_bound, max_bound, 1000).T  # Correct shape
    pdf_vals = norm.pdf(x_vals, loc=0, scale=sigma_L[:, None]) + norm.pdf(x_vals, loc=0, scale=sigma_R[:, None])
    pdf_vals /= pdf_vals.sum(axis=1, keepdims=True)
    cdf_vals = np.cumsum(pdf_vals, axis=1) / cdf_vals[:, -1][:, None]

    random_probs = np.random.uniform(0, 1, size)
    return np.array([np.interp(p, cdf, x) for p, cdf, x in zip(random_probs, cdf_vals, x_vals)])

# ─────────────────────────────────────────────────────────────────────────
# LIFT DATA SIMULATION
# ─────────────────────────────────────────────────────────────────────────
def simulate_iteration(data_chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate actual reps and weights for a given dataset.
    """
    data_chunk = data_chunk[data_chunk['Assigned Weight'] > 0].copy()
    r_assigned, w_assigned = data_chunk['# of Reps'].values, data_chunk['Assigned Weight'].values

    data_chunk['Actual Reps'] = r_assigned + generate_rep_differences_vectorized(r_assigned)
    data_chunk['Actual Weight'] = w_assigned + np.random.normal(loc=0, scale=0.1 * w_assigned, size=len(w_assigned))
    return data_chunk
