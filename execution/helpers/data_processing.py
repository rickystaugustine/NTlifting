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

    # ✅ Dynamically determine core lifts from program_df
    if "Relevant Core" not in program_df.columns:
        raise KeyError("❌ ERROR: 'Relevant Core' column is missing in program_df!")

    core_lifts = program_df["Relevant Core"].dropna().unique().tolist()

    # ✅ Ensure required columns exist dynamically
    required_cols = {"Player"} | set(core_lifts)
    missing_cols = required_cols - set(core_maxes_df.columns)
    if missing_cols:
        raise KeyError(f"❌ ERROR: Missing columns in core_maxes_df: {missing_cols}")

    logging.info(f"📊 Core maxes dataframe shape: {core_maxes_df.shape}")

    # ✅ Convert core maxes into a melted format dynamically
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
    
    # ✅ Ensure Relevant Core is dynamically assigned
    if "Relevant Core" not in repeated_program_df.columns:
        logging.info("🔍 Extracting 'Relevant Core' mapping dynamically from CompleteProgram...")
    
        # Find unique (Exercise, Relevant Core) pairs
        core_mapping = program_df[["Exercise", "Relevant Core"]].drop_duplicates()
    
        if core_mapping["Relevant Core"].isnull().any():
            logging.warning("⚠️ Some exercises have no 'Relevant Core' mapping! Defaulting to 'Unknown'.")
    
        logging.info(f"✅ Extracted {len(core_mapping)} unique Exercise ↔ Relevant Core mappings.")
    
        # ✅ Merge dynamically extracted mapping
        repeated_program_df = repeated_program_df.merge(core_mapping, on="Exercise", how="left")
        repeated_program_df["Relevant Core"].fillna("Unknown", inplace=True)
        logging.info("✅ 'Relevant Core' successfully assigned dynamically.")
    
    # ✅ Assertions to catch issues earlier
    assert "Relevant Core" in repeated_program_df.columns, "❌ ERROR: 'Relevant Core' column missing after merge!"
    assert "Relevant Core" in flattened_core_maxes_df.columns, "❌ ERROR: 'Relevant Core' column missing in core maxes!"

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
