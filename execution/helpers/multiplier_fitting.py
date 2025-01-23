import logging
from helpers.exercise_fitting import fit_exercise_multipliers
import numpy as np
from scipy.optimize import curve_fit

def fit_multipliers(repeated_program_df):
    """Fits multipliers for all exercises separately from weight assignment."""
    logging.info("Fitting multipliers for exercises...")
    exercise_functions = fit_exercise_multipliers(repeated_program_df)
    
    logging.info(f"Fitted multipliers for {len(exercise_functions)} exercises.")
    return exercise_functions

def rep_to_percentage(reps, a, b, c):
    """Exponential function mapping reps to percentage of max."""
    return a * np.exp(-b * reps) + c

def fit_percentage_curve(program_df):
    """Fits a rep-to-percentage curve using non-linear least squares regression."""
    grouped = program_df.groupby('# of Reps')['Multiplier of Max'].mean().reset_index()
    reps = grouped['# of Reps'].values
    percentages = grouped['Multiplier of Max'].values
    
    try:
        popt, _ = curve_fit(rep_to_percentage, reps, percentages, maxfev=2000)
        print("Fitted Equation Parameters: a={}, b={}, c={}".format(*popt))
        return lambda reps: rep_to_percentage(reps, *popt)
    except RuntimeError as e:
        print(f"Curve fitting failed: {e}")
        # Fall back to simpler polynomial fit if needed
        coefficients = np.polyfit(reps, percentages, deg=2)
        poly_function = np.poly1d(coefficients)
        print(f"Using fallback polynomial fit: coefficients={coefficients}")
        return lambda reps: poly_function(reps)

