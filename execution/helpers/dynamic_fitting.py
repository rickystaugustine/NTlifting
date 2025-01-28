import numpy as np

def dynamic_multiplier_adjustment(r_assigned, r_actual, M_assigned):
    """
    Adjust the multiplier dynamically based on the difference between 
    assigned reps (r_assigned) and actual reps (r_actual).
    
    Args:
    r_assigned (int): Assigned repetitions.
    r_actual (int): Actual repetitions performed.
    M_assigned (float): Assigned multiplier for the exercise.
    
    Returns:
    float: Adjusted multiplier (M_actual).
    """
    
    # Calculate the rep difference ratio
    rep_diff_ratio = r_actual / r_assigned

    # Adjust multiplier based on rep difference ratio
    # This formula assumes a linear adjustment, but can be modified for more complex behavior
    M_actual = M_assigned * np.exp((rep_diff_ratio - 1) * 0.05)  # Adjust this scaling factor as needed

    # Ensure that the multiplier doesn't go below a threshold
    M_actual = max(M_actual, 0.1)  # Prevents M_actual from being too small

    return M_actual

