import sys
import os
import pickle
import pandas as pd
import numpy as np  # Ensure NumPy is imported
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Dynamically add the NTlifting root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)

# Import necessary functions
from execution.helpers.data_processing import preprocess_data

def merge_data(repeated_program_df, flattened_core_maxes_df, multiplier_fits):
    """ Merges expanded program data with core maxes and applies multipliers. """

    logging.info("🔄 Running merge_data() function...")

    # Perform the merge
    merged_df = repeated_program_df.merge(flattened_core_maxes_df, on=["Player", "Relevant Core"], how="left")

    # Fix: Check for duplicate `Tested Max` columns and rename properly
    if "Tested Max_x" in merged_df.columns and "Tested Max_y" in merged_df.columns:
        logging.warning("⚠️ Detected duplicate `Tested Max` columns! Resolving issue...")

        # Use the `Tested Max_y` column and drop the other
        merged_df["Tested Max"] = merged_df["Tested Max_y"].fillna(merged_df["Tested Max_x"])
        merged_df.drop(columns=["Tested Max_x", "Tested Max_y"], inplace=True)

    logging.info("✅ Merged successfully, 'Tested Max' available in merged_data.")

    return merged_df
