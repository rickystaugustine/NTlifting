import pandas as pd
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from execution.helpers.google_sheets_utils import read_google_sheets, write_to_google_sheet
from execution.helpers.exercise_fitting import fit_single_exercise_global
from execution.helpers.simulation import simulate_iteration
from execution.helpers.multipliers import ConstantMultiplier, FittedMultiplier
import sys
import os

# Ensure execution/ is in Python's module search path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/helpers"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from execution.helpers.data_loading import load_data

from execution.helpers.dynamic_fitting import dynamic_multiplier_adjustment

def calculate_functional_maxes(simulated_data):
    logging.info("Calculating functional maxes...")

    # Iterate through the simulated data to handle each case
    for index, row in simulated_data.iterrows():
        r_assigned = row['# of Reps']  # Assigned reps
        r_actual = row['Actual Reps']  # Actual reps
        W_assigned = row['Assigned Weight']  # Assigned weight
        W_actual = row['Actual Weight']  # Actual weight

        # Debugging: Show the data for each row
        logging.debug(f"Row {index} - r_assigned: {r_assigned}, r_actual: {r_actual}, W_assigned: {W_assigned}, W_actual: {W_actual}")

        # Case 1: r_actual ≠ r_assigned && W_actual = W_assigned
        if r_actual != r_assigned and W_actual == W_assigned:
            logging.debug(f"Row {index} falls into Case 1")
            # Adjust the multiplier dynamically based on the actual vs assigned reps
            M_assigned = row['Multiplier']  # Assigned multiplier
            M_actual = dynamic_multiplier_adjustment(r_assigned, r_actual, M_assigned)
            
            # Calculate Functional Max using the adjusted multiplier
            functional_max = W_actual / M_actual
            simulated_data.at[index, 'Functional Max'] = functional_max

        # Case 2: r_actual = r_assigned && W_actual ≠ W_assigned
        elif r_actual == r_assigned and W_actual != W_assigned:
            # logging.debug(f"Row {index} falls into Case 2")
            # Calculate Functional Max by dividing actual weight by the assigned multiplier
            M_assigned = row['Multiplier']  # Assigned multiplier
            functional_max = W_actual / M_assigned
            simulated_data.at[index, 'Functional Max'] = functional_max

        # Case 3: r_actual ≠ r_assigned && W_actual ≠ W_assigned
        elif r_actual != r_assigned and W_actual != W_assigned:
            # logging.debug(f"Row {index} falls into Case 3")
            # Handle Case 3 with combined iterative method (not implemented here)
            pass

    logging.info("Functional maxes calculated successfully.")
    return simulated_data
