import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_multiweek_projections(player, exercise_code, current_week, current_functional_max, sma, ewma, linear_trend,
                                    num_cycles=4, growth_rate_cap_per_cycle=0.02, decay_factor=0.95):
    """
    Generate multi-week Functional Max projections for a player/exercise.
    
    Args:
        player (str): Player identifier.
        exercise_code (int): Exercise identifier.
        current_week (int): Current week number.
        current_functional_max (float): Latest Functional Max estimate.
        sma (float): Simple Moving Average trend estimate.
        ewma (float): Exponential Weighted Moving Average trend estimate.
        linear_trend (float): Linear regression trend estimate.
        num_cycles (int): How many 3-week cycles to project.
        growth_rate_cap_per_cycle (float): Max percentage Functional Max can increase per cycle.
        decay_factor (float): Factor to dampen aggressive projections.

    Returns:
        list of dicts: Each dict contains projected Functional Max by cycle.
    """

    projections = []
    base_projection = current_functional_max

    # Calculate an average trend as baseline (you can tweak this)
    trend_baseline = (sma + ewma + linear_trend) / 3

    for cycle in range(1, num_cycles + 1):
        # Conservative growth factor capped by growth_rate_cap_per_cycle
        projected_increase = min(
            (trend_baseline - current_functional_max) * decay_factor,
            current_functional_max * growth_rate_cap_per_cycle
        )

        base_projection += projected_increase

        projections.append({
            "Player": player,
            "Exercise Code": exercise_code,
            "Cycle #": cycle,
            "Projected Functional Max": round(base_projection, 2)
        })

    # logging.info(f"Generated {num_cycles} cycle projections for {player} - Exercise {exercise_code}. "
      #           f"Start Functional Max: {current_functional_max:.2f}, "
       #          f"End Projection: {base_projection:.2f}")

    return projections
