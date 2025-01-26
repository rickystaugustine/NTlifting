import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_DIR = "data"  # Ensure this directory exists

def preprocess_data(program_df, core_maxes_df):
    logging.info("🚀 Running preprocess_data()...")

    logging.info("🔄 Preprocessing core maxes...")
    core_lifts = ["Bench", "Barbell Squat", "Clean", "Hex-Bar Deadlift"]

    logging.info(f"📊 Core maxes dataframe shape: {core_maxes_df.shape}")
  
    # Convert core maxes into a melted format
    flattened_core_maxes_df = core_maxes_df.melt(
        id_vars=["Player"], 
        value_vars=core_lifts, 
        var_name="Relevant Core", 
        value_name="Tested Max"
    )

    # Convert 'Tested Max' column efficiently
    flattened_core_maxes_df["Tested Max"] = (
        flattened_core_maxes_df["Tested Max"]
        .astype(str).str.strip()
        .replace("", "NRM")  # Ensure true empty values are replaced
    )
    
    # Convert numeric values while keeping "NRM" as a string
    flattened_core_maxes_df["Tested Max"] = pd.to_numeric(flattened_core_maxes_df["Tested Max"], errors="coerce").fillna("NRM")
    logging.info(f"✅ Flattened core maxes shape: {flattened_core_maxes_df.shape}")

    logging.info("🔄 Expanding workout program for all players...")
    
    # Ensure 'Player' column exists
    if "Player" not in core_maxes_df.columns or core_maxes_df["Player"].isnull().all():
        logging.error("❌ ERROR: 'Player' column is missing or empty in maxes data.")
        return pd.DataFrame(), pd.DataFrame()
    
    players_df = core_maxes_df[["Player"]].drop_duplicates()
    repeated_program_df = players_df.merge(program_df, how="cross")
    
    logging.info(f"✅ Repeated program generated with {len(repeated_program_df)} records.")

    # Save the processed data to pickle files
    os.makedirs(DATA_DIR, exist_ok=True)

    repeated_program_path = os.path.join(DATA_DIR, "repeated_program.pkl")
    flattened_core_maxes_path = os.path.join(DATA_DIR, "flattened_core_maxes.pkl")

    logging.info("💾 Saving processed data...")
    repeated_program_df.to_pickle(repeated_program_path)
    flattened_core_maxes_df.to_pickle(flattened_core_maxes_path)
    
    logging.info(f"✅ Saved repeated program to {repeated_program_path}")
    logging.info(f"✅ Saved flattened core maxes to {flattened_core_maxes_path}")

    return flattened_core_maxes_df, repeated_program_df
