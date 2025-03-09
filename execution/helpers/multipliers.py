import pandas as pd
import numpy as np
import pickle
import os
import logging

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

# Log the first few items in multiplier_fits before function creation
# logging.info(f"DEBUG: First few items in multiplier_fits: {list(multiplier_fits.items())[:10]}")

# Generate exercise functions from stored coefficients
exercise_functions = {}
for code, coeffs in multiplier_fits.items():
    if isinstance(coeffs, list):  # Ensure coefficients are stored as a list
        exercise_functions[code] = create_multiplier_function(coeffs)
        # logging.info(f"DEBUG: {code} -> {coeffs}")
    else:
        exercise_functions[code] = lambda w, s, r, c=coeffs: c  # Wrap static values in a function

# Save the updated multiplier functions
with open(multipliers_path, "wb") as f:
    pickle.dump(multiplier_fits, f)

# logging.info(f"Successfully saved multiplier_fits.pkl at {multipliers_path}.")

# Ensure exercise_functions is available for import
__all__ = ["multiplier_fits", "exercise_functions"]

class ConstantMultiplier:
    """ A simple constant multiplier class. """
    def __init__(self, value):
        self.value = value
        self.coefficients = [value]  # Store as a list for consistency

class FittedMultiplier:
    """ A fitted multiplier that applies dynamic coefficients. """
    def __init__(self, params):
        self.params = params
        self.coefficients = params  # Ensure coefficients are directly accessible

# Define m_func as a wrapper for multiplier functions
def m_func(inputs, *params):
    w, s, r = inputs
    if len(params) < 4:
        raise ValueError(f"❌ ERROR: Expected at least 4 parameters but got {len(params)}")
    return params[0] * w + params[1] * s + params[2] * np.log(r + 1) + params[3]

# Ensure these functions are available for import
__all__ = ["ConstantMultiplier", "FittedMultiplier", "m_func", "multiplier_fits", "exercise_functions"]
