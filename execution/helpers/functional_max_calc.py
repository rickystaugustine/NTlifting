import pandas as pd
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from helpers.google_sheets_utils import read_google_sheets, write_to_google_sheet
from helpers.exercise_fitting import fit_single_exercise_global
from helpers.simulation import simulate_iteration
from helpers.multipliers import ConstantMultiplier, FittedMultiplier
import sys
import os

# Ensure execution/ is in Python's module search path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/helpers"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from helpers.data_loading import load_data

def calculate_functional_maxes(simulated_data):
    logging.info("Calculating functional maxes...")
    simulated_data["Functional Max"] = simulated_data["Actual Weight"] / simulated_data["Multiplier"]
    functional_maxes = simulated_data.groupby(["Player", "Relevant Core"])["Functional Max"].median().reset_index()
    logging.info("Functional maxes calculated successfully.")
    return functional_maxes
