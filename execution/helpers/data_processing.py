import pandas as pd
import logging

def preprocess_data(program_df, core_maxes_df):
    logging.info("Preprocessing core maxes...")
    core_lifts = ["Bench", "Barbell Squat", "Clean", "Hex-Bar Deadlift"]
    flattened_core_maxes_df = core_maxes_df.melt(id_vars=["Player"], value_vars=core_lifts, var_name="Relevant Core", value_name="Tested Max")
    logging.info("Creating repeated program...")
    players = core_maxes_df["Player"].unique()
    repeated_program_df = pd.concat([program_df.assign(Player=player) for player in players], ignore_index=True)
    
    print("🔍 Checking DataFrames before merging...")
    print("Columns in repeated_program_df:", list(repeated_program_df.columns))
    print("Columns in flattened_core_maxes_df:", list(flattened_core_maxes_df.columns))
    print("First few rows of repeated_program_df:")
    print(repeated_program_df.head())
    print("First few rows of flattened_core_maxes_df:")
    print(flattened_core_maxes_df.head())
    
    logging.info(f"Repeated program generated with {len(repeated_program_df)} records.")
    return flattened_core_maxes_df, repeated_program_df
