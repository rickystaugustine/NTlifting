import pandas as pd
import numpy as np
import logging
import sys
import os
import pickle
from sklearn.linear_model import LinearRegression
from execution.helpers.google_sheets_utils import write_to_google_sheet
from execution.helpers.multiplier_calc import (
    is_constant_multiplier,
    calculate_adjusted_multiplier_function,
    calculate_adjusted_multiplier_iterative
)
from execution.helpers.trend_estimation import generate_trend_estimates
from execution.helpers.evaluation import evaluate_functional_max_adjustment
from execution.helpers.data_processing import preprocess_expanded_df
from execution.helpers.plots import generate_aggregate_plots
from execution.helpers.reports import generate_summary_stats

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load the multiplier fits directly from the .pkl file
def load_multiplier_fits():
    """Loads the multiplier_fits.pkl file from the explicitly defined path."""
    pkl_path = "/Users/ricky.staugustine/Documents/Programming/GitHub/NTlifting/execution/helpers/multiplier_fits.pkl"  # ✅ Explicit correct path

    try:
        with open(pkl_path, "rb") as file:
            multiplier_fits = pickle.load(file)
        return multiplier_fits
    except Exception as e:
        logging.error(f"❌ Error loading multiplier_fits.pkl from {pkl_path}: {e}")
        return {}


def calculate_functional_max(row):
    """Calculates Functional Max based on Method."""
    if row["Method"] == "Ratio":
        if row["Tested Max"] > 0 and row["Assigned Weight"] > 0:
            return row["Tested Max"] * (row["Simulated Weight"] / row["Assigned Weight"])
        else:
            return None  # Prevents division errors
    return None  # Placeholder for other methods



def assign_cases(expanded_df, multiplier_fits):
    """Assigns Cases, Multiplier Type, Method, and Adjusted Multiplier."""
    # Preprocess and validate the input DataFrame
    expanded_df = preprocess_expanded_df(expanded_df)
    expanded_df["Case"] = 3  # Default all to Case 3
    expanded_df["Multiplier Type"] = "Fitted"  # Default all to Fitted
    expanded_df["Method"] = "Iterative"  # Default all to Iterative
    expanded_df["Adjusted Multiplier"] = None  # Placeholder for Adjusted Multiplier

    expanded_df = expanded_df.reset_index(drop=True)
    # Define filtered exercises and initialize trend summary for CSV report
    filtered_exercises = [1, 2, 14]  # Barbell Squat, Barbell Front Squat, Hex-Bar Deadlift (example codes)
    trend_summary = []

    # Calculate Reps Match and Weights Close before assigning cases
    expanded_df["Reps Match"] = expanded_df.apply(
        lambda row: int(row["Simulated Reps"]) == int(row["# of Reps"]), axis=1
    )
    expanded_df["Weights Close"] = expanded_df.apply(
        lambda row: abs(float(row["Simulated Weight"]) - float(row["Assigned Weight"])) < 1, axis=1
    )

    expanded_df.loc[(expanded_df["Reps Match"]) & (~expanded_df["Weights Close"]), "Case"] = 1
    expanded_df.loc[(~expanded_df["Reps Match"]) & (expanded_df["Weights Close"]), "Case"] = 2

    # Determine if the exercise uses a Constant or Fitted multiplier using Exercise Code
    expanded_df["Multiplier Type"] = expanded_df["Code"].apply(
        lambda code: "Constant" if is_constant_multiplier(code, multiplier_fits
    ) else "Fitted")

    # Assign "Method" Based on Case and Multiplier Type
    expanded_df.loc[expanded_df["Case"] == 1, "Method"] = "Ratio"
    expanded_df.loc[(expanded_df["Case"] == 2) & (expanded_df["Multiplier Type"] == "Constant"), "Method"] = "Scale"
    expanded_df.loc[(expanded_df["Case"] == 2) & (expanded_df["Multiplier Type"] == "Fitted"), "Method"] = "Function"

    # Ensure "Multiplier of Max" Column Exists Before Assigning Adjusted Multiplier
    if "Multiplier of Max" in expanded_df.columns:
        # Assign "Adjusted Multiplier" Based on Method
        mask_ratio = expanded_df["Method"] == "Ratio"
        expanded_df.loc[mask_ratio, "Adjusted Multiplier"] = expanded_df.loc[mask_ratio, "Multiplier of Max"].values

        mask_function = expanded_df["Method"] == "Function"
        expanded_df.loc[mask_function, "Adjusted Multiplier"] = expanded_df[mask_function].apply(
            lambda row: calculate_adjusted_multiplier_function(row, multiplier_fits, k=10),  # Adjust k if needed
            axis=1
        )

        mask_scale = expanded_df["Method"] == "Scale"
        expanded_df.loc[mask_scale, "Adjusted Multiplier"] = (
            expanded_df.loc[mask_scale, "Multiplier of Max"].values *
            (expanded_df.loc[mask_scale, "# of Reps"].values /
             expanded_df.loc[mask_scale, "Simulated Reps"].values)
        )

        mask_iterative = expanded_df["Method"] == "Iterative"
        expanded_df.loc[mask_iterative, "Adjusted Multiplier"] = expanded_df[mask_iterative].apply(
            lambda row: calculate_adjusted_multiplier_iterative(row, multiplier_fits, k=10),
            axis=1
        )
    else:
        logging.error("❌ ERROR: 'Multiplier of Max' column is missing from expanded_df.")

    # Calculate "Functional Max" Based on Method
    expanded_df.loc[expanded_df["Method"] == "Ratio", "Functional Max"] = (
        expanded_df["Tested Max"].astype(np.float64) *
        (expanded_df["Simulated Weight"].astype(np.float64) / expanded_df["Assigned Weight"].astype(np.float64))
    )
    expanded_df.loc[expanded_df["Method"] == "Function", "Functional Max"] = (
        expanded_df["Simulated Weight"].astype(np.float64) /
        expanded_df["Adjusted Multiplier"].astype(np.float64)
    )
    expanded_df.loc[expanded_df["Method"] == "Scale", "Functional Max"] = (
        expanded_df.loc[expanded_df["Method"] == "Scale", "Simulated Weight"].astype(np.float64) /
        expanded_df.loc[expanded_df["Method"] == "Scale", "Adjusted Multiplier"].astype(np.float64)
    )
    expanded_df.loc[expanded_df["Method"] == "Iterative", "Functional Max"] = (
        expanded_df.loc[expanded_df["Method"] == "Iterative", "Simulated Weight"].astype(np.float64) /
        expanded_df.loc[expanded_df["Method"] == "Iterative", "Adjusted Multiplier"].astype(np.float64)
    )

    # Log trend estimates for each athlete/exercise/week combination and collect CSV data
    for idx, row in expanded_df.iterrows():
        # Ensure required columns exist; adjust column names if necessary
        if "Player" in row and "Week #" in row and "Code" in row:
            athlete_id = row["Player"]
            exercise_code = row["Code"]
            current_week = row["Week #"]

            trend_estimates = generate_trend_estimates(
                expanded_df,
                athlete_id,
                exercise_code,
                current_week,
                sma_window=3,
                ewma_alpha=0.5
            )

            if (trend_estimates["SMA"] is not None and
                trend_estimates["EWMA"] is not None and
                trend_estimates["Linear Trend"] is not None):

                # Collect trend estimates for CSV output
                trend_summary.append({
                    "Player": athlete_id,
                    "Exercise Code": exercise_code,
                    "Week": current_week,
                    "SMA": trend_estimates["SMA"],
                    "EWMA": trend_estimates["EWMA"],
                    "Linear Trend": trend_estimates["Linear Trend"],
                    "Functional Max": row["Functional Max"]
                })

    # Convert to numeric, fill missing values, and explicitly cast to np.float64
    expanded_df["Adjusted Multiplier"] = pd.to_numeric(expanded_df["Adjusted Multiplier"], errors="coerce").fillna(0).astype(np.float64)
    expanded_df["Functional Max"] = pd.to_numeric(expanded_df["Functional Max"], errors="coerce").fillna(0).astype(np.float64)

    # Ensure that Functional Max is not missing before upload
    if "Functional Max" not in expanded_df.columns:
        logging.error("❌ ERROR: 'Functional Max' column is missing from expanded_df after assignment.")

    expanded_df["Functional Max"] = expanded_df["Functional Max"].astype(np.float64)

    # Add Strength Change column
    expanded_df["Strength Change"] = expanded_df.apply(
        lambda row: +1 if (row["Functional Max"] - row["Tested Max"]) > 1 else
                    -1 if (row["Functional Max"] - row["Tested Max"]) < -1 else
                    0,
        axis=1
    )

    # Add Expected Change column using the evaluation function
    expanded_df["Expected Change"] = expanded_df.apply(
        lambda row: evaluate_functional_max_adjustment(row, multiplier_fits), axis=1
    )

    # Add Mismatch column to identify discrepancies
    expanded_df["Mismatch"] = expanded_df["Strength Change"] != expanded_df["Expected Change"]

    generate_summary_stats(expanded_df)

    trend_summary_df = pd.DataFrame(trend_summary)
    return expanded_df, trend_summary_df
