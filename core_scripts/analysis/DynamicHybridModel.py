import numpy as np
from scipy.optimize import curve_fit, minimize
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt

# Define the cubic and hybrid-log-exp models
def cubic_model(reps, a, b, c, d):
    return a * reps**3 + b * reps**2 + c * reps + d

def hybrid_log_exp_model(reps, a, b, c, d, e):
    return a * np.log(reps + b) + c * np.exp(-d * reps) + e

# Cross-validation to calculate MAE
def cross_validate_model(model_function, reps, percentages, params):
    splitter = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    maes = []

    for train_idx, test_idx in splitter.split(reps):
        reps_train, reps_test = reps[train_idx], reps[test_idx]
        percentages_train, percentages_test = percentages[train_idx], percentages[test_idx]
        
        predictions = model_function(reps_test, *params)
        mae = np.mean(np.abs(percentages_test - predictions))
        maes.append(mae)

    return np.mean(maes)

# Residuals plotting function
def plot_residuals(model_function, reps, percentages, params, title):
    predictions = model_function(reps, *params)
    residuals = percentages - predictions
    
    plt.figure(figsize=(6, 8))
    plt.scatter(reps, residuals, label="Residuals")
    plt.axhline(0, color="red", linestyle="--", label="Zero Line")
    plt.title(f"Residuals Plot for {title}")
    plt.xlabel("Reps")
    plt.ylabel("Residuals")
    plt.legend()
    plt.show()

# Dynamic parameter adjustment for hybrid-log-exp model
def refine_hybrid_model_parameters(reps, percentages):
    initial_guesses = [
        np.float64(0.2),  # Initial guess for 'a'
        0.1,              # Initial guess for 'b'
        np.float64(0.01), # Initial guess for 'c'
        0.1,              # Initial guess for 'd'
        np.float64(0.7),  # Initial guess for 'e'
    ]

    bounds = [
        (0, np.inf),   # Bounds for 'a'
        (0, np.inf),   # Bounds for 'b'
        (-np.inf, np.inf),  # Bounds for 'c'
        (0, np.inf),   # Bounds for 'd'
        (-np.inf, np.inf)   # Bounds for 'e'
    ]

    def residual_objective(params):
        predictions = hybrid_log_exp_model(reps, *params)
        residuals = percentages - predictions
        return np.sum(residuals**2)  # Minimize the sum of squared residuals

    result = minimize(
        residual_objective,
        initial_guesses,
        bounds=bounds,
        method='L-BFGS-B'
    )

    if result.success:
        print(f"Hybrid model optimization succeeded: {result.message}")
        return result.x
    else:
        print(f"Hybrid model failed to converge: {result.message}")
        return None

# Preprocess and Normalize Data
reps = np.array([3, 4, 5, 6, 7])
percentages = np.array([0.9, 0.85, 0.8, 0.75, 0.7])

# Normalize the data (ensure no division by zero or invalid range)
reps_range = reps.max() - reps.min()
percentages_range = percentages.max() - percentages.min()

if reps_range == 0 or percentages_range == 0:
    raise ValueError("Normalization failed due to zero range.")

reps_scaled = (reps - reps.min()) / reps_range
percentages_scaled = (percentages - percentages.min()) / percentages_range

# Debugging to ensure data is correct
print("Reps Scaled:", reps_scaled)
print("Percentages Scaled:", percentages_scaled)

# Ensure no missing or NaN values in normalized arrays
if np.isnan(reps_scaled).any() or np.isnan(percentages_scaled).any():
    raise ValueError("NaN values found in normalized data.")

# Fit the Cubic Model
popt_cubic, _ = curve_fit(cubic_model, reps_scaled, percentages_scaled, maxfev=10000)
cubic_mae = cross_validate_model(cubic_model, reps_scaled, percentages_scaled, popt_cubic)
print(f"Cubic Model MAE: {cubic_mae:.4f}")

# Fit the Hybrid Log-Exp Model
try:
    popt_hybrid = refine_hybrid_model_parameters(reps_scaled, percentages_scaled)
    if popt_hybrid is not None:
        hybrid_mae = cross_validate_model(hybrid_log_exp_model, reps_scaled, percentages_scaled, popt_hybrid)
        print(f"Hybrid_log_exp Model MAE: {hybrid_mae:.4f}")
except Exception as e:
    print(f"Error fitting hybrid-log-exp model: {e}")
    popt_hybrid, hybrid_mae = None, float("inf")
  
# Select the Best Model
if popt_hybrid is not None and hybrid_mae < cubic_mae:
    best_model = "Hybrid_log_exp"
    best_mae = hybrid_mae
    best_params = popt_hybrid
else:
    best_model = "Cubic"
    best_mae = cubic_mae
    best_params = popt_cubic

print(f"Best Model Selected: {best_model} with MAE: {best_mae:.4f}")

# Enhanced Residuals Plotting
plot_residuals(
    cubic_model if best_model == "Cubic" else hybrid_log_exp_model,
    reps_scaled,
    percentages_scaled,
    best_params,
    best_model
)
# Denormalize residuals for interpretation in the original scale
def denormalize_residuals(residuals, percentages_min, percentages_range):
    return residuals * percentages_range + percentages_min

# Calculate additional metrics (RMSE, R²)
def calculate_metrics(model_function, reps, percentages, params):
    predictions = model_function(reps, *params)
    residuals = percentages - predictions
    rmse = np.sqrt(np.mean(residuals**2))
    ss_total = np.sum((percentages - np.mean(percentages))**2)
    ss_residual = np.sum(residuals**2)
    r2 = 1 - (ss_residual / ss_total)
    return rmse, r2

# Denormalize residuals and calculate metrics
denormalized_residuals = denormalize_residuals(percentages_scaled - cubic_model(reps_scaled, *popt_cubic), percentages.min(), percentages_range)
rmse, r2 = calculate_metrics(cubic_model, reps_scaled, percentages_scaled, popt_cubic)

print(f"Denormalized Residuals: {denormalized_residuals}")
print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Store model parameters and MAE results
results = {
    "Cubic": {"params": popt_cubic, "mae": cubic_mae},
    "Hybrid_log_exp": {"params": popt_hybrid, "mae": hybrid_mae if popt_hybrid is not None else None},
}

# Save residuals and results to a file for detailed analysis
import json

output_data = {
    "Residuals": denormalized_residuals.tolist(),
    "Metrics": {"RMSE": rmse, "R²": r2},
    "Model Results": {
        "Best Model": best_model,
        "Best MAE": best_mae,
        "Cubic Params": popt_cubic.tolist(),
        "Hybrid Params": popt_hybrid.tolist() if popt_hybrid is not None else None,
    }
}

with open("model_analysis.json", "w") as f:
    json.dump(output_data, f, indent=4)

print("Model analysis saved to 'model_analysis.json'.")

# Automate comparison chart between models
def plot_comparison(reps, percentages, cubic_params, hybrid_params=None):
    plt.figure(figsize=(10, 6))
    
    # Original Data
    plt.scatter(reps, percentages, color="blue", label="Original Data")
    
    # Cubic Model
    cubic_predictions = cubic_model(reps, *cubic_params)
    plt.plot(reps, cubic_predictions, color="green", label="Cubic Model")
    
    # Hybrid Model (if available)
    if hybrid_params is not None:
        hybrid_predictions = hybrid_log_exp_model(reps, *hybrid_params)
        plt.plot(reps, hybrid_predictions, color="orange", label="Hybrid Log-Exp Model")
    
    plt.title("Model Comparisons")
    plt.xlabel("Reps")
    plt.ylabel("Percentages")
    plt.legend()
    plt.show()

# Execute comparison plot
plot_comparison(reps_scaled, percentages_scaled, popt_cubic, popt_hybrid if popt_hybrid is not None else None)
