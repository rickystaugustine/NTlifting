import sys
import os
import logging
import pandas as pd

# Ensure execution/ is in Python's module search path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Try importing preprocess_data with fallback in case of import issues
try:
    from execution.helpers.google_sheets_utils import read_google_sheets
    from execution.helpers.data_processing import preprocess_data
except ModuleNotFoundError:
    logging.error("❌ ERROR: Failed to import 'preprocess_data'. Retrying with adjusted sys.path.")
    sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../helpers"))
    from execution.helpers.data_processing import preprocess_data

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_data():
    """Loads raw program and maxes data from Google Sheets without preprocessing."""
    try:
        logging.info("📥 Loading raw data from Google Sheets...")
        program_df = read_google_sheets("After-School Lifting", "CompleteProgram")
        core_maxes_df = read_google_sheets("After-School Lifting", "Maxes")
        
        logging.info(f"✅ Loaded {len(program_df)} rows from CompleteProgram")
        logging.info(f"✅ Loaded {len(core_maxes_df)} rows from Maxes")
        
        return program_df, core_maxes_df
    except Exception as e:
        logging.error(f"❌ ERROR: Failed to load data from Google Sheets: {e}")
        return pd.DataFrame(), pd.DataFrame()  # Return empty DataFrames on failure

def load_and_preprocess_data(preprocess=True):
    """Loads and optionally preprocesses program and maxes data from Google Sheets."""
    program_df, core_maxes_df = load_data()
    
    if preprocess:
        logging.info("🔄 Preprocessing data...")
        if program_df.empty or core_maxes_df.empty:
            logging.warning("⚠️ WARNING: One or both DataFrames are empty before preprocessing.")
        
        try:
            flattened_core_maxes_df, repeated_program_df = preprocess_data(program_df, core_maxes_df)
            
            # Fix Pandas FutureWarning: Explicitly set the 'Tested Max' column
            flattened_core_maxes_df["Tested Max"] = flattened_core_maxes_df["Tested Max"].fillna("NRM")
            
            logging.info("✅ Data preprocessing complete.")
            return flattened_core_maxes_df, repeated_program_df
        except Exception as e:
            logging.error(f"❌ ERROR: Failed during data preprocessing: {e}")
            return pd.DataFrame(), pd.DataFrame()  # Return empty DataFrames on failure
    
    return program_df, core_maxes_df
