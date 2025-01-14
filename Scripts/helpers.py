import warnings
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import norm
from scipy.integrate import quad

def m_func(inputs, a, b, c, d):
    w, s, r = inputs
    return a * w + b * s + c * np.log(r + 1) + d


class ConstantMultiplier:
    """
    A globally defined class for constant multiplier.
    """
    def __init__(self, value):
        self.value = value

    def __call__(self, w, s, r):
        return self.value


class FittedMultiplier:
    """
    A globally defined class for fitted multiplier.
    """
    def __init__(self, params):
        self.params = params

    def __call__(self, w, s, r):
        return m_func((w, s, r), *self.params)


def fit_single_exercise_global(code, program_records):
    """
    Fit a multiplier function for a single exercise.
    """
    program_df = pd.DataFrame(program_records)  # Deserialize records
    exercise_df = program_df[program_df['Code'] == code]
    weeks = exercise_df['Week #'].values
    sets = exercise_df['Set #'].values
    reps = exercise_df['# of Reps'].values
    multipliers = exercise_df['Multiplier of Max'].values

    if len(set(multipliers)) == 1:
        print(f"Exercise {code} has a constant multiplier: {multipliers[0]}")
        return code, ConstantMultiplier(multipliers[0])  # Return an instance of ConstantMultiplier

    initial_guess = [1, 1, 1, multipliers.mean()]
    bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=OptimizeWarning)
            popt, _ = curve_fit(m_func, (weeks, sets, reps), multipliers, p0=initial_guess, bounds=bounds, maxfev=10000)

        return code, FittedMultiplier(popt)  # Return an instance of FittedMultiplier

    except (OptimizeWarning, RuntimeError) as e:
        print(f"Fit failed for exercise {code} ({e}). Using mean multiplier.")
        return code, ConstantMultiplier(multipliers.mean())  # Return an instance of ConstantMultiplier

def generate_rep_differences_vectorized(r_assigned_array):
    """
    Generate rep differences for an array of assigned reps using a custom PDF.

    Parameters:
        r_assigned_array (np.ndarray): Array of assigned reps.

    Returns:
        np.ndarray: Array of rep differences.
    """
    size = len(r_assigned_array)

    min_bound = -0.9 * r_assigned_array
    max_bound = 3.0 * r_assigned_array

    sigma_L = ((-1 * min_bound) * 0.997) / 3
    sigma_R = (max_bound * 0.997) / 3

    # Pre-generate x values for each assigned rep
    x_vals = np.linspace(min_bound, max_bound, 1000).T  # Correct the shape to (12510, 1000)

    def combined_pdf_vectorized(x, r, sigma_L, sigma_R, min_bound, max_bound):
        """
        Combined PDF for left and right tails, vectorized and normalized.
        
        Parameters:
            x (np.ndarray): Array of x values (broadcasted).
            r (np.ndarray): Assigned reps (broadcasted to match x).
            sigma_L (np.ndarray): Std deviation for the left side (broadcasted to match x).
            sigma_R (np.ndarray): Std deviation for the right side (broadcasted to match x).
            min_bound (np.ndarray): Minimum bound (broadcasted to match x).
            max_bound (np.ndarray): Maximum bound (broadcasted to match x).

        Returns:
            np.ndarray: PDF values for the entire array.
        """
        # Ensure proper broadcasting
        x, r, sigma_L, sigma_R, min_bound, max_bound = np.broadcast_arrays(
            x, r[:, None], sigma_L[:, None], sigma_R[:, None], min_bound[:, None], max_bound[:, None]
        )

        pdf = np.zeros_like(x)  # Initialize PDF array with zeros

        # Masks for left and right sides
        left_mask = (x < 0) & (x >= min_bound)
        right_mask = (x >= 0) & (x <= max_bound)

        # Calculate PDF values for left and right masks
        pdf[left_mask] = norm.pdf(x[left_mask], loc=0, scale=sigma_L[left_mask])
        pdf[right_mask] = norm.pdf(x[right_mask], loc=0, scale=sigma_R[right_mask])

        return pdf

    # Calculate the PDF values for all x values
    pdf_vals = combined_pdf_vectorized(x_vals, r_assigned_array, sigma_L, sigma_R, min_bound, max_bound)

    # Normalize the PDF
    normalization_constants = pdf_vals.sum(axis=1, keepdims=True)
    normalized_pdf_vals = pdf_vals / normalization_constants

    # Generate random samples
    cdf_vals = np.cumsum(normalized_pdf_vals, axis=1)
    cdf_vals /= cdf_vals[:, -1][:, None]  # Normalize CDF

    random_probs = np.random.uniform(0, 1, size)
    # Interpolate sampled values
    sampled_vals = np.array([
        np.interp(random_prob, cdf_row, x_row)
        for random_prob, cdf_row, x_row in zip(random_probs, cdf_vals, x_vals)
    ])

    return sampled_vals

def simulate_iteration(data_chunk):
    data_chunk = data_chunk[data_chunk['Assigned Weight'] > 0]
    r_assigned = data_chunk['# of Reps'].values
    w_assigned = data_chunk['Assigned Weight'].values

    rep_differences = generate_rep_differences_vectorized(r_assigned)
    data_chunk['Actual Reps'] = r_assigned + rep_differences

    weight_differences = np.random.normal(loc=0, scale=0.1 * w_assigned, size=len(w_assigned))
    data_chunk['Actual Weight'] = w_assigned + weight_differences

    return data_chunk
