import pandas as pd
import logging
import os
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_DIR = "data"  # Ensure this directory exists

def preprocess_data(program_df, maxes_df):
    """Preprocess program and maxes data."""

    logging.info("🚀 Running preprocess_data()...")

    # Ensure missing values in Maxes are replaced with 'NRM'
    maxes_df.fillna("NRM", inplace=True)

    # Flatten core maxes for merging
    flattened_core_maxes_df = maxes_df.melt(id_vars=["Player"], var_name="Relevant Core", value_name="Tested Max")
    logging.info(f"✅ Flattened core maxes shape: {flattened_core_maxes_df.shape}")

    # Expand program_df to apply to every player
    num_players = maxes_df["Player"].nunique()
    expanded_program_df = program_df.loc[program_df.index.repeat(num_players)].reset_index(drop=True)

    # Add Player column by repeating all players for each row
    expanded_program_df["Player"] = np.tile(maxes_df["Player"].values, len(program_df))

    logging.info(f"✅ Expanded program generated with {len(expanded_program_df)} records.")

    # 🚨 Fix: Merge `Tested Max` early so it's available for multipliers
    merged_data = expanded_program_df.merge(flattened_core_maxes_df, on=["Player", "Relevant Core"], how="left")

    logging.info(f"✅ Merged Tested Max into repeated program data. Columns now: {list(merged_data.columns)}")

    return flattened_core_maxes_df, merged_data
