import pandas as pd
import numpy as np
import pickle
import os

def create_multiplier_function(params):
    def multiplier_function(w, s, r, coeffs=params):  # Accept coeffs properly
        return coeffs[0] * w + coeffs[1] * s + coeffs[2] * np.log(r + 1) + coeffs[3]
    return multiplier_function

# Load the preprocessed data
multipliers_path = os.path.join(os.path.dirname(__file__), "multiplier_fits.pkl")

if os.path.exists(multipliers_path):
    with open(multipliers_path, "rb") as f:
        multiplier_fits = pickle.load(f)
else:
    multiplier_fits = {}

# Generate exercise functions from stored coefficients
exercise_functions = {}
for code, coeffs in multiplier_fits.items():
    if isinstance(coeffs, list):  # Ensure coefficients are stored as a list
        exercise_functions[code] = create_multiplier_function(coeffs)
    else:
        exercise_functions[code] = lambda w, s, r, c=coeffs: c  # Wrap static values in a function

# Save the updated multiplier functions
with open(multipliers_path, "wb") as f:
    pickle.dump(multiplier_fits, f)

# Ensure exercise_functions is available for import
__all__ = ["multiplier_fits", "exercise_functions"]
