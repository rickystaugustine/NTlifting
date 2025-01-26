import sys
import os
import logging
import pandas as pd

# Ensure execution/ is in Python's module search path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Try importing preprocess_data with fallback in case of import issues
try:
    from execution.helpers.data_loading import load_data, load_and_preprocess_data
except ModuleNotFoundError:
    logging.error("❌ ERROR: Failed to import 'preprocess_data'. Retrying with adjusted sys.path.")
    sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../helpers"))
    from helpers.data_processing import preprocess_data

# Set up logging
logging.basicConfig(level=logging.INFO)

# ✅ Step 1: Test raw data loading
print("\n📥 Testing raw data loading...")
program_df, core_maxes_df = load_data()

if program_df.empty or core_maxes_df.empty:
    print("❌ ERROR: DataFrames are empty. Check Google Sheets connection.")
else:
    print(f"✅ Loaded Program Data: {len(program_df)} rows")
    print(f"✅ Loaded Maxes Data: {len(core_maxes_df)} rows")
    print(program_df.head())  # Display first 5 rows
    print(core_maxes_df.head())

# ✅ Step 2: Test data preprocessing
print("\n🔄 Testing data preprocessing...")
flattened_core_maxes_df, repeated_program_df = load_and_preprocess_data()

if flattened_core_maxes_df.empty or repeated_program_df.empty:
    print("❌ ERROR: Preprocessed DataFrames are empty. Check preprocessing function.")
else:
    print(f"✅ Flattened Core Maxes Data: {len(flattened_core_maxes_df)} rows")
    print(f"✅ Repeated Program Data: {len(repeated_program_df)} rows")
    print(flattened_core_maxes_df.head())  # Display first 5 rows
    print(repeated_program_df.head())
