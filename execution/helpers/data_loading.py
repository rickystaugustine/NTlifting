import sys
import os

# Ensure execution/ is in Python's module search path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from helpers.google_sheets_utils import read_google_sheets
from helpers.data_processing import preprocess_data

def load_data():
    """Loads raw program and maxes data from Google Sheets without preprocessing."""
    return read_google_sheets("After-School Lifting", "CompleteProgram"), read_google_sheets("After-School Lifting", "Maxes")

def load_and_preprocess_data():
    """Loads and preprocesses program and maxes data from Google Sheets."""
    program_df, core_maxes_df = load_data()
    flattened_core_maxes_df, repeated_program_df = preprocess_data(program_df, core_maxes_df)
    return flattened_core_maxes_df, repeated_program_df
