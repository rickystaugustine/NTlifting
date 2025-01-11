import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.optimize import curve_fit

def rep_to_percentage(reps, a, b, c):
    """Exponential function to calculate percentage of max based on reps."""
    return a * np.exp(-b * reps) + c

def fit_high_hang_pull():
    """Fits the High-Hang Pull model using exponential or linear regression as a fallback."""
    data = {
        '# of Reps': [3, 4, 5],
        'Multiplier of Max': [0.847, 0.770, 0.700]
    }
    df = pd.DataFrame(data)
    reps = df['# of Reps'].values
    percentages = df['Multiplier of Max'].values

    try:
        weights = 1 / (percentages + 1e-6)
        popt, _ = curve_fit(rep_to_percentage, reps, percentages, sigma=weights, maxfev=10000, p0=[0.5, 0.1, 0.5])
        print("Exponential fit successful for High-Hang Pull")

        def high_hang_pull_func(reps):
            return rep_to_percentage(reps, *popt)

        return high_hang_pull_func

    except RuntimeError as e:
        print(f"High-Hang Pull fitting failed: {e}")
        coefficients = np.polyfit(reps, percentages, deg=1)
        return np.poly1d(coefficients)

def simulate_actual_lifts(program_df, maxes_df, n_variants=100):
    """Generates simulated data for Actual Lifts."""
    simulated_data = []

    for _, row in program_df.iterrows():
        exercise = row['Exercise']
        multiplier = row['Multiplier of Max']
        reps = row['# of Reps']

        for _, player in maxes_df.iterrows():
            tested_max = player[row['Relevant Core']]
            if not pd.isna(tested_max):
                assigned_weight = tested_max * multiplier

                for _ in range(n_variants):
                    actual_weight = assigned_weight + np.random.uniform(-5, 5)
                    actual_reps = reps + np.random.choice([-1, 0, 1])
                    simulated_data.append({
                        'Player': player['Player'],
                        'Exercise': exercise,
                        'Actual Weight': actual_weight,
                        'Actual Reps': actual_reps,
                        'Tested Max': tested_max
                    })

    return pd.DataFrame(simulated_data)

def calculate_functional_max(simulated_data, percentage_functions):
    """Calculates Functional Max using fitted percentage functions."""
    functional_max_results = []

    for _, row in simulated_data.iterrows():
        exercise = row['Exercise']
        actual_weight = row['Actual Weight']
        actual_reps = row['Actual Reps']

        percentage_function = percentage_functions.get(exercise, lambda reps: 1)
        actual_percentage = percentage_function(actual_reps)
        functional_max = actual_weight / actual_percentage

        functional_max_results.append({
            'Player': row['Player'],
            'Exercise': exercise,
            'Actual Weight': actual_weight,
            'Actual Reps': actual_reps,
            'Functional Max': round(functional_max, 2),
            'Tested Max': row['Tested Max']
        })

    return pd.DataFrame(functional_max_results)

def evaluate_model(functional_max_results):
    """Evaluates the overall model accuracy and residuals."""
    mae = mean_absolute_error(functional_max_results['Tested Max'], functional_max_results['Functional Max'])
    r2 = r2_score(functional_max_results['Tested Max'], functional_max_results['Functional Max'])

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R^2): {r2:.2f}")

    residuals = functional_max_results['Functional Max'] - functional_max_results['Tested Max']
    plt.scatter(functional_max_results['Tested Max'], residuals)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Tested Max")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.show()

    return mae, r2

def main():
    """Main function to execute the testing process."""
    # Mock program and maxes data
    program_data = pd.DataFrame({
        'Exercise': ["High-Hang Pull", "Bench", "Squat"],
        'Relevant Core': ["Clean", "Bench", "Squat"],
        '# of Reps': [3, 5, 7],
        'Multiplier of Max': [0.847, 0.7, 0.85]
    })

    maxes_data = pd.DataFrame({
        'Player': ["Player1", "Player2", "Player3"],
        'Bench': [250, 200, 300],
        'Squat': [300, 350, 400],
        'Clean': [200, 220, 180]
    })

    # Generate simulated lifts
    simulated_data = simulate_actual_lifts(program_data, maxes_data)

    # Fit High-Hang Pull function
    percentage_functions = {
        "High-Hang Pull": fit_high_hang_pull(),
        "Bench": lambda reps: 1 - 0.02 * reps,
        "Squat": lambda reps: 1 - 0.015 * reps
    }

    # Calculate Functional Max
    functional_max_results = calculate_functional_max(simulated_data, percentage_functions)

    # Evaluate model
    mae, r2 = evaluate_model(functional_max_results)
    print("Evaluation Complete.")

if __name__ == "__main__":
    main()
