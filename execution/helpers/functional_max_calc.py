import numpy as np
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def saturation_function(r_as, r_ac, k=1.2, c=0.2):
    """
    Limits extreme rep increases using a sigmoid-like function.

    - r_as: Assigned reps
    - r_ac: Actual reps executed
    - k: Max allowed multiplier adjustment
    - c: Sensitivity factor (higher = more resistant to change)
    
    Returns:
    - Adjustment factor for Functional Max calculation.
    """
    return k / (1 + np.exp(-c * (r_ac - r_as)))

def update_functional_max(prev_xf, M_ac, alpha=0.2):
    """
    Uses a weighted moving average to smooth Functional Max changes.

    - prev_xf: Previous Functional Max
    - M_ac: Newly calculated Functional Max adjustment
    - alpha: Weighting factor (higher = faster updates, lower = smoother updates)
    
    Returns:
    - Smoothed Functional Max estimate.
    """
    return alpha * M_ac + (1 - alpha) * prev_xf

def calculate_functional_max(prev_xf, tested_max, assigned_weight, actual_weight, assigned_reps, actual_reps, beta=1.4, gamma=0.15, decay_factor=0.1):
    """
    Computes Functional Max using a weight-dominant adjustment model.
    """
    # If first calculation, start with tested max to prevent overestimation
    if prev_xf is None:
        prev_xf = tested_max

    # Increase weight impact
    weight_ratio = (actual_weight / assigned_weight) ** beta
    
    # Reduce rep sensitivity
    rep_adjustment = (actual_reps / assigned_reps) ** gamma

    # Apply rep-effort decay
    rep_decay = np.exp(-decay_factor * abs(actual_reps - assigned_reps))
    adjusted_rep_ratio = rep_adjustment * rep_decay

    adjustment_factor = weight_ratio * adjusted_rep_ratio

    # More aggressive drops for weight reductions
    if prev_xf * adjustment_factor < prev_xf:
        alpha = 0.5  # Faster updates for declines
    else:
        alpha = 0.2  # Slower updates for increases

    updated_xf = alpha * (prev_xf * adjustment_factor) + (1 - alpha) * prev_xf
    
    return round(updated_xf, 2)
