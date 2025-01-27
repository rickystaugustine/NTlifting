import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_DIR = "data"  # Ensure this directory exists

def preprocess_data(program_df, core_maxes_df):
    """Prepares the training data by expanding the workout program for each player
       and converting core maxes into a structured format.
    """
    logging.info("🚀 Running preprocess_data()...")

    core_lifts = ["Bench", "Barbell Squat", "Clean", "Hex-Bar Deadlift"]
    required_cols = {"Player"} | set(core_lifts)

    # ✅ Ensure required columns exist
    missing_cols = required_cols - set(core_maxes_df.columns)
    if missing_cols:
        raise KeyError(f"❌ ERROR: Missing columns in core_maxes_df: {missing_cols}")

    logging.info(f"📊 Core maxes dataframe shape: {core_maxes_df.shape}")

    # ✅ Convert core maxes into a melted format
    flattened_core_maxes_df = core_maxes_df.melt(
        id_vars=["Player"],
        value_vars=core_lifts,
        var_name="Relevant Core",
        value_name="Tested Max"
    )

    # ✅ Standardize 'Tested Max' column
    flattened_core_maxes_df["Tested Max"] = (
        pd.to_numeric(flattened_core_maxes_df["Tested Max"], errors="coerce")
        .fillna("NRM")
    )

    logging.info(f"✅ Flattened core maxes shape: {flattened_core_maxes_df.shape}")

    # ✅ Expand the workout program for all players
    if core_maxes_df["Player"].isnull().all():
        logging.error("❌ ERROR: 'Player' column is empty in maxes data.")
        return pd.DataFrame(), pd.DataFrame()

    repeated_program_df = core_maxes_df[["Player"]].drop_duplicates().merge(program_df, how="cross")
    
    logging.info(f"✅ Repeated program generated with {len(repeated_program_df)} records.")

    # ✅ Save the processed data
    os.makedirs(DATA_DIR, exist_ok=True)

    repeated_program_path = os.path.join(DATA_DIR, "repeated_program.pkl")
    flattened_core_maxes_path = os.path.join(DATA_DIR, "flattened_core_maxes.pkl")

    logging.info("💾 Saving processed data...")
    repeated_program_df.to_pickle(repeated_program_path)
    flattened_core_maxes_df.to_pickle(flattened_core_maxes_path)

    logging.info(f"✅ Processed data saved to {DATA_DIR}/")
    
    return flattened_core_maxes_df, repeated_program_df
