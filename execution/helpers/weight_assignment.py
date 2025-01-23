import pandas as pd
import logging
import numpy as np

def assign_weights(repeated_program_df, flattened_core_maxes_df, exercise_functions):
    logging.info("Calculating multipliers and assigned weights...")
    repeated_program_df = repeated_program_df.merge(flattened_core_maxes_df, on=["Player", "Relevant Core"], how="left")
    repeated_program_df["Tested Max"] = pd.to_numeric(repeated_program_df["Tested Max"], errors="coerce").fillna(0)
    weeks, sets, reps, codes = repeated_program_df["Week #"].values, repeated_program_df["Set #"].values, repeated_program_df["# of Reps"].values, repeated_program_df["Code"].values
    multipliers = [exercise_functions[int(code)][1](w, s, r) for code, w, s, r in zip(codes, weeks, sets, reps)]
    repeated_program_df["Multiplier"] = multipliers
    repeated_program_df["Assigned Weight"] = (np.floor(repeated_program_df["Tested Max"] * repeated_program_df["Multiplier"] / 5) * 5)
    logging.info(f"Assigned weights calculated for {len(repeated_program_df)} records.")
    return repeated_program_df
