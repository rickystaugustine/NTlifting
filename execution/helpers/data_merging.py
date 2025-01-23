import pandas as pd

def merge_data(program_df, maxes_df):
    """Merges program data with maxes data to compute calculated weights."""
    core_to_column = {
        'Bench': 'Bench',
        'Squat': 'Squat',
        'Clean': 'Clean',
        'Deadlift': 'Deadlift'
    }
    expanded_rows = []

    for _, exercise_row in program_df.iterrows():
        relevant_core = exercise_row['Relevant Core']
        multiplier = exercise_row['Multiplier of Max']

        if relevant_core in core_to_column:
            max_column = core_to_column[relevant_core]

            for _, player_row in maxes_df.iterrows():
                if pd.notna(player_row[max_column]):
                    try:
                        max_lift = float(player_row[max_column]) if player_row[max_column] else 0.0
                    except ValueError:
                        max_lift = 0.0  # Default to 0 if conversion fails

                    calculated_weight = max_lift * multiplier

                    rounded_weight = round(calculated_weight / 5) * 5
                    expanded_row = exercise_row.copy()
                    expanded_row['Player'] = player_row['Player']
                    expanded_row['Max Lift'] = player_row[max_column]
                    expanded_row['Calculated Weight'] = rounded_weight
                    expanded_rows.append(expanded_row)

    return pd.DataFrame(expanded_rows)

