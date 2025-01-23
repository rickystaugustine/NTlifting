import numpy as np
import pandas as pd
from scipy.stats import norm
import logging

def generate_rep_differences_vectorized(r_assigned_array: np.ndarray) -> np.ndarray:
    """
    Generate rep differences based on a probability distribution.
    """
    size = len(r_assigned_array)
    min_bound, max_bound = -0.9 * r_assigned_array, 5.0 * r_assigned_array
    sigma_L, sigma_R = ((-1 * min_bound) * 0.997) / 3, (max_bound * 0.997) / 3

    x_vals = np.linspace(min_bound, max_bound, 1000).T  # Correct shape

    # Compute the PDF values
    pdf_vals = norm.pdf(x_vals, loc=0, scale=sigma_L[:, None]) + norm.pdf(x_vals, loc=0, scale=sigma_R[:, None])

    # Normalize the PDF to avoid division by zero
    pdf_sums = pdf_vals.sum(axis=1, keepdims=True)
    pdf_sums[pdf_sums == 0] = 1  # Avoid division errors
    pdf_vals /= pdf_sums

    # Compute the CDF
    cdf_vals = np.cumsum(pdf_vals, axis=1)
    
    # Ensure last value is never zero before normalization
    cdf_vals[:, -1][cdf_vals[:, -1] == 0] = 1  
    cdf_vals /= cdf_vals[:, -1][:, None]

    # Sample values using inverse CDF sampling
    random_probs = np.random.uniform(0, 1, size)
    return np.array([np.interp(p, cdf, x) for p, cdf, x in zip(random_probs, cdf_vals, x_vals)])

def simulate_iteration(data_chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate actual reps and weights for a given dataset.
    """
    data_chunk = data_chunk[data_chunk['Assigned Weight'] > 0].copy()
    r_assigned, w_assigned = data_chunk['# of Reps'].values, data_chunk['Assigned Weight'].values

    data_chunk['Actual Reps'] = r_assigned + generate_rep_differences_vectorized(r_assigned)
    data_chunk['Actual Weight'] = w_assigned + np.random.normal(loc=0, scale=0.1 * w_assigned, size=len(w_assigned))
    return data_chunk

def simulate_lift_data(repeated_program_df, num_iterations=5):
    logging.info("Simulating lift data...")
    simulated_data = pd.concat([simulate_iteration(repeated_program_df) for _ in range(num_iterations)], ignore_index=True)
    logging.info(f"Simulated data generated with {len(simulated_data)} records.")
    return simulated_data
