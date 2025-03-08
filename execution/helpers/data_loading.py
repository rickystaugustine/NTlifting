import sys
import os
import logging
import pandas as pd

# Try importing preprocess_data with fallback in case of import issues
try:
    from execution.helpers.google_sheets_utils import read_google_sheets
    from execution.helpers.data_processing import preprocess_data
except ModuleNotFoundError:
    logging.error("❌ ERROR: Failed to import 'preprocess_data'. Retrying with adjusted sys.path.")
    sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../helpers"))

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_data():
    """Loads CompleteProgram and Maxes data from Google Sheets."""
    
    # logging.info("📥 Loading raw data from Google Sheets...")

    # Retrieve Google Sheets Data
    program_df = read_google_sheets("After-School Lifting", "CompleteProgram")
    maxes_df = read_google_sheets("After-School Lifting", "Maxes")

    # logging.info(f"✅ Loaded {len(program_df)} rows from CompleteProgram")
    # logging.info(f"✅ Loaded {len(maxes_df)} rows from Maxes")

    # Ensure "Player" column is properly named
    if "player" in program_df.columns:
        program_df.rename(columns={"player": "Player"}, inplace=True)
    if "player" in maxes_df.columns:
        maxes_df.rename(columns={"player": "Player"}, inplace=True)

    return program_df, maxes_df
