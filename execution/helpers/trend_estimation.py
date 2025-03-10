import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import logging

def estimate_sma(history_df, window=3):
    """Simple Moving Average of Functional Max over a specified window."""
    if history_df.empty or len(history_df) < window:
        return None
    sma = history_df["Functional Max"].rolling(window=window).mean().iloc[-1]
    return sma

def estimate_ewma(history_df, alpha=0.5):
    """Exponentially Weighted Moving Average (EWMA) for Functional Max."""
    if history_df.empty:
        return None
    ewma = history_df["Functional Max"].ewm(alpha=alpha).mean().iloc[-1]
    return ewma

def estimate_linear_trend(history_df, current_week):
    """Linear Regression trend estimate for Functional Max."""
    if history_df.empty or len(history_df) < 2:
        return None

    X = history_df["Week #"].values.reshape(-1, 1)
    y = history_df["Functional Max"].values

    model = LinearRegression().fit(X, y)

    # Explicitly convert current_week to a float (or int)
    week_numeric = float(current_week)

    trend_estimate = model.predict(np.array([[week_numeric]]))[0]
    return trend_estimate

def generate_trend_estimates(history_df, athlete_id, exercise_code, current_week, sma_window=3, ewma_alpha=0.5):
    """Generate multiple trend-based Functional Max estimates."""
    athlete_history = history_df[
        (history_df["Player"] == athlete_id) &
        (history_df["Code"] == exercise_code) &
        (history_df["Week #"] < current_week)
    ].sort_values(by="Week #")

    if athlete_history.empty:
        default_trend = history_df[
            (history_df["Player"] == athlete_id) &
            (history_df["Code"] == exercise_code)
        ]["Tested Max"].iloc[0]

        return {
            "SMA": default_trend,
            "EWMA": default_trend,
            "Linear Trend": default_trend
        }

    return {
        "SMA": estimate_sma(athlete_history, window=sma_window),
        "EWMA": estimate_ewma(athlete_history, alpha=ewma_alpha),
        "Linear Trend": estimate_linear_trend(athlete_history, current_week)
    }
