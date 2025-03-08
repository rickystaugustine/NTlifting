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

class ConstantMultiplier:
    """ A simple constant multiplier class. """
    def __init__(self, value):
        self.value = value

    def apply(self, w, s, r):
        return self.value

class FittedMultiplier:
    """ A fitted multiplier that applies dynamic coefficients. """
    def __init__(self, params):
        self.params = params

    def apply(self, w, s, r):
        return self.params[0] * w + self.params[1] * s + self.params[2] * np.log(r + 1) + self.params[3]

# Define m_func as a wrapper for multiplier functions
def m_func(inputs, *params):
    w, s, r = inputs
    if len(params) < 4:
        raise ValueError(f"❌ ERROR: Expected at least 4 parameters but got {len(params)}")
    return params[0] * w + params[1] * s + params[2] * np.log(r + 1) + params[3]

# Ensure these functions are available for import
__all__ = ["ConstantMultiplier", "FittedMultiplier", "m_func", "multiplier_fits", "exercise_functions"]
