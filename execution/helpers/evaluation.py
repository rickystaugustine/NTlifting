import numpy as np

def exp_decay(reps, a, b, c):
    return a * np.exp(-b * reps) + c

import logging

def evaluate_functional_max_adjustment(row, multiplier_fits, margin = 0.005):
    """
    Evaluates whether Functional Max should increase, decrease, or stay the same based on 
    the per-exercise empirical reps vs weight curve.
    """
    try:
        tested_max = float(row["Tested Max"])
        assigned_reps = float(row["# of Reps"])
        assigned_weight = float(row["Assigned Weight"])
        simulated_reps = float(row["Simulated Reps"])
        simulated_weight = float(row["Simulated Weight"])
        exercise_code = row["Code"]
    except Exception as e:
        logging.error(f"❌ Error converting row values for evaluation: {e}")
        return 0

    coeffs = multiplier_fits.get(exercise_code)

    a_fit, b_fit, c_fit, _ = coeffs[:4]

    expected_relative_weight_sim = exp_decay(simulated_reps, a_fit, b_fit, c_fit)
    assigned_relative_weight = assigned_weight / tested_max
    simulated_relative_weight = simulated_weight / tested_max

    # Fixed logic: comparison using a margin
    if simulated_relative_weight > assigned_relative_weight + margin:
        return +1
    elif simulated_relative_weight < assigned_relative_weight - margin:
        return -1
    else:
        return 0
