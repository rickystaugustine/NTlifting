import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, r2_score

def rep_to_percentage(reps, a, b, c):
    """
    Exponential function to calculate percentage of max based on reps.
    """
    return a * np.exp(-b * reps) + c

def fit_high_hang_pull():
    """
    Fits the High-Hang Pull model using provided data and returns the percentage function.
    """
    print("Fitting curve for High-Hang Pull")

    # Provided data
    data = {
        "# of Reps": [3, 4, 5],
        "Multiplier of Max": [0.847, 0.770, 0.700]
    }
    
    df = pd.DataFrame(data)
    reps = df["# of Reps"].values
    percentages = df["Multiplier of Max"].values

    try:
        # Fit exponential curve
        weights = 1 / (percentages + 1e-6)
        popt, _ = curve_fit(rep_to_percentage, reps, percentages, sigma=weights, maxfev=10000, p0=[0.5, 0.1, 0.5])

        def high_hang_pull_percentage(reps):
            return rep_to_percentage(reps, *popt)

        print("Exponential fit successful for High-Hang Pull")
        print("Predicted multipliers for High-Hang Pull:", [high_hang_pull_percentage(r) for r in range(3, 7)])

        return high_hang_pull_percentage

    except RuntimeError as e:
        print(f"Curve fitting failed for High-Hang Pull: {e}")
        print("Falling back to linear fit")
        coefficients = np.polyfit(reps, percentages, deg=1)
        fallback_model = np.poly1d(coefficients)

        def high_hang_pull_percentage_linear(reps):
            return fallback_model(reps)

        return high_hang_pull_percentage_linear

def integrate_high_hang_pull_fix(existing_functions):
    """
    Integrates the High-Hang Pull fix into the existing percentage functions.
    """
    high_hang_pull_function = fit_high_hang_pull()
    existing_functions["High-Hang Pull"] = high_hang_pull_function
    return existing_functions

def calculate_functional_max(simulated_data, percentage_functions):
    """
    Calculates Functional Max values using the updated percentage functions.
    """
    results = []

    for _, row in simulated_data.iterrows():
        exercise = row['Exercise']
        actual_weight = row['Actual Weight']
        actual_reps = row['Actual Reps']

        percentage_function = percentage_functions.get(exercise, lambda reps: 1)
        actual_percentage = percentage_function(actual_reps)
        functional_max = actual_weight / actual_percentage

        results.append({
            'Player': row['Player'],
            'Exercise': exercise,
            'Actual Weight': actual_weight,
            'Actual Reps': actual_reps,
            'Functional Max': round(functional_max, 2)
        })

    return pd.DataFrame(results)

def main():
    # Placeholder for existing functions (to be replaced with your actual implementation)
    percentage_functions = {
        "1-Arm DB Row": lambda reps: 0.1375,
        "3-Way DB Shoulder Raise": lambda reps: 0.125,
        "Bar RDL": lambda reps: 0.25,
        "High-Hang Pull": None  # To be updated
    }

    # Simulate some data (replace with actual simulated data)
    simulated_data = pd.DataFrame({
        'Player': ['Player1', 'Player2', 'Player3'],
        'Exercise': ['High-Hang Pull', 'High-Hang Pull', 'High-Hang Pull'],
        'Actual Weight': [200, 210, 190],
        'Actual Reps': [3, 4, 5]
    })

    # Integrate High-Hang Pull fix
    percentage_functions = integrate_high_hang_pull_fix(percentage_functions)

    # Calculate Functional Max
    functional_max_results = calculate_functional_max(simulated_data, percentage_functions)
    print(functional_max_results)

if __name__ == "__main__":
    main()
