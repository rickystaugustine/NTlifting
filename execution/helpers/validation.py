import numpy as np
import pandas as pd

def validate_multiplier_fits(complete_program_df, multiplier_fits):
    results = []
    
    for _, row in complete_program_df.iterrows():
        exercise_code = row['Code']
        week_num = row['Week #']
        set_num = row['Set #']
        assigned_reps = row['# of Reps']
        expected_multiplier = row['Multiplier of Max']
        
        coeffs = multiplier_fits.get(exercise_code)
        if not coeffs:
            logging.warning(f"No coeffs found for Exercise Code {exercise_code}")
            continue
        
        A, B, C, D = coeffs[:4]

        # Calculate predicted multiplier
        k = assigned_reps * 1.2  # Or whatever makes sense based on your decay tuning
        k = max(k, assigned_reps + 1)

        predicted_multiplier = (
            A * week_num +
            B * set_num +
            C * np.log(k / (assigned_reps + 1)) +
            D
        )

        results.append({
            "Exercise Code": exercise_code,
            "Week #": week_num,
            "Set #": set_num,
            "Assigned Reps": assigned_reps,
            "Expected Multiplier": expected_multiplier,
            "Predicted Multiplier": predicted_multiplier,
            "Difference": predicted_multiplier - expected_multiplier
        })
    
    return pd.DataFrame(results)
