import numpy as np
import logging

def is_constant_multiplier(exercise_code, multiplier_fits):
    """Detects if an exercise is a constant multiplier by checking its stored coefficients using Exercise Code."""
    try:
        if exercise_code not in multiplier_fits:
            return False  # Assume it's a fitted multiplier if not found

        coeffs = multiplier_fits[exercise_code]

        # Check if first 3 coefficients are insignificant
        nonzero_coeffs = [c for c in coeffs[:3] if abs(c) > 1e-6]

        is_constant = len(nonzero_coeffs) == 0
        return is_constant

    except Exception as e:
        logging.warning(f"⚠️ Error checking constant multiplier for Exercise Code '{exercise_code}': {e}")

    return False


def calculate_adjusted_multiplier_function(row, multiplier_fits, k=10):
    exercise_code = row["Code"]

    try:
        week_num = float(row["Week #"])
        set_num = float(row["Set #"])
        sim_reps = float(row["Simulated Reps"])
    except Exception as e:
        logging.error(f"❌ Error converting row values to float for exercise code {exercise_code}: {e}")
        return 0

    coeffs = multiplier_fits.get(exercise_code)
    A, B, C, D = coeffs[:4]

    try:
        adj_multiplier = (
            A * week_num +
            B * set_num +
            C * np.log(k / (sim_reps + 1)) +
            D
        )
        return adj_multiplier
    except Exception as e:
        logging.error(f"❌ Error calculating Adjusted Multiplier for exercise code {exercise_code}: {e}")
        return 0


def calculate_adjusted_multiplier_iterative(row, multiplier_fits, k=10):
    exercise_code = row["Code"]

    try:
        week_num = float(row["Week #"])
        set_num = float(row["Set #"])
        sim_reps = float(row["Simulated Reps"])
        sim_weight = float(row["Simulated Weight"])
        assigned_reps = float(row["# of Reps"])
        assigned_weight = float(row["Assigned Weight"])
        tested_max = float(row["Tested Max"])
    except Exception as e:
        logging.error(f"❌ Error converting row values for exercise code {exercise_code}: {e}")
        return 0

    coeffs = multiplier_fits.get(exercise_code)
    if coeffs is None:
        logging.error(f"❌ No coefficients found for exercise code {exercise_code}.")
        return 0

    A, B, C, D = coeffs[:4]

    try:
        expected_multiplier = (
            A * week_num +
            B * set_num +
            C * np.log(k / (sim_reps + 1)) +
            D
        )
    except Exception as e:
        logging.error(f"❌ Error calculating expected multiplier for exercise code {exercise_code}: {e}")
        return 0

    if tested_max == 0:
        logging.error(f"❌ Tested Max is zero for exercise code {exercise_code}. Cannot calculate expected weight.")
        return 0

    min_expected_multiplier = (assigned_weight * 0.75) / tested_max if tested_max > 0 else 0
    expected_multiplier = max(expected_multiplier, min_expected_multiplier)

    expected_weight = expected_multiplier * tested_max

    if expected_weight == 0:
        logging.error(f"❌ Expected weight is zero for exercise code {exercise_code}. Cannot calculate adjusted multiplier.")
        return 0

    adj_multiplier = sim_weight / expected_weight

    return adj_multiplier
