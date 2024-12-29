import pandas as pd

# Specify the columns you want to read
columns_to_read = ['Exercise', 'e', 'r', 's', 'w']

# Load only the specified columns from the CSV file
df = pd.read_csv('/Users/ricky.staugustine/Documents/FB/After-School Lifting - Weights.csv', usecols=columns_to_read)

# Convert the DataFrame to a list of tuples
exercise_data = list(df.itertuples(index=False, name=None))

def add_player(name, bench, squat, clean, deadlift):
    players_max_cores[name] = {'Bench': bench, 'Squat': squat, 'Clean': clean, 'Deadlift': deadlift}

# Dictionary to Store Player Maximums
players_max_cores = {}
add_player('Johnson', 250, 400, 175, 435)
add_player('Pearson', 155, 295, 125, 300)
add_player('Ott', 135, 250, 115, 245)
add_player('Marker', 135, 340, 160, 295)
add_player('Clark', 180, 300, 165, 330)
add_player('Wolfe', 85, 95, 85, 135)
# Add more players as needed

def get_player_max_core(player, e):
    # Map from 'e' values to corresponding lifts
    if 1 <= e < 4:
        core = 'Squat'
    elif 4 <= e < 6:
        core = 'Clean'
    elif 6 <= e < 12:
        core = 'Bench'
    elif 12 <= e < 15:
        core = 'Deadlift'
    else:
        return f"Invalid value of e: {e}. e must be between 1 and 14."

    # Get the player's max for the determined core lift
    player_max = players_max_cores.get(player, {}).get(core)
    
    if player_max is not None:
        return f"{player}'s maximum for {core} is {player_max} lbs."
    else:
        return f"No data available for {player} or core lift {core}."

# Example
# print(get_player_max_core('Johnson',2))

def P(e, r, s, w):
    if e == 1:
        return 0.0066482 * r + 0.0617036 * s + 0.03054017 * w + 0.5209833795013851
    elif e == 2:
        return 0.003241 * r + 0.030518 * s + 0.01527008 * w + 0.26049168975069253
    elif e == 3:
        return 0.075
    elif e == 4:
        return -0.03675 * r + 0.03675 * w + 0.845833333333333
    elif e == 5:
        return 0.01239669 * r + 0.1125 * s + 0.03533058 * w + 0.42004132231404967
    elif e == 6:
        return 0.00698643 * r + 0.06477715 * s + 0.03207202 * w + 0.5469977839335181
    elif e == 7:
        return 0.0066482 * r + 0.0617036 * s + 0.03054017 * w + 0.5209833795013851
    elif e == 8:
        return 0.1375
    elif e == 9:
        return 0.00265014 * r + 0.04158306 * s + 0.02095845 * w + 0.326088457987073
    elif e == 10:
        return 0.125
    elif e == 11:
        return 0.125
    elif e == 12:
        return 0.25
    elif e == 13:
        return 0.4
    elif e == 14:
        return 0.0066482 * r + 0.0617036 * s + 0.03054017 * w + 0.5209833795013851
    else:
        return None

def W(player, e, r, s, w):
    core_lift = {
        1: 'Squat', 2: 'Squat', 3: 'Squat', 4: 'Clean', 5: 'Clean',
        6: 'Bench', 7: 'Bench', 8: 'Bench', 9: 'Bench', 10: 'Bench', 11: 'Bench',
        12: 'Deadlift', 13: 'Deadlift', 14: 'Deadlift'
    }.get(e)

    relevant_core_max = players_max_cores.get(player, {}).get(core_lift)
    if relevant_core_max is None:
        return "Invalid data or configuration"

    percentage_of_relevant_core_max = P(e, r, s, w)
    weight = percentage_of_relevant_core_max * relevant_core_max
    rounded_weight = int((weight // 5) * 5)  # Ensure the weight is rounded and converted to int here
    return rounded_weight

# Test the function
# print(W('Johnson', 1, 5, 1, 1))

# Function to display all exercise weights for all players and collect the data
def display_exercise_weights():
    all_data = []  # List to collect all the data
    for player in players_max_cores.keys():
        for exercise, e, r, s, w in exercise_data:
            weight_output = W(player, e, r, s, w)
            # Append each result as a dictionary to the list
            all_data.append({
                "Player": player,
                "Exercise": exercise,
                "Week": w,
                "Set": s,
                "Reps": r,
                "Weight [lb]": weight_output  # No need for conversion here
            })
            # print(f"{player}: {exercise} - Week {w}, Set {s}, Reps {r} => {weight_output} lbs")
    return all_data

# Execute the function and store the results
exercise_results = display_exercise_weights()

# Create a DataFrame from your data
df = pd.DataFrame(exercise_results)

# Path for the Excel file
excel_path = 'grouped_exercise_data.xlsx'

# Create a Pandas Excel writer using XlsxWriter as the engine.
with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    # Iterate over each unique player in the DataFrame
    for player in df['Player'].unique():
        # Filter data for the current player
        player_data = df[df['Player'] == player]
        
        # Write each player's data to a separate sheet
        player_data.to_excel(writer, sheet_name=player, index=False)

with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    workbook = writer.book
    for player in df['Player'].unique():
        player_data = df[df['Player'] == player]
        player_data.to_excel(writer, sheet_name=player, index=False)
        
        # Get the xlsxwriter worksheet object
        worksheet = writer.sheets[player]
        
        # Set the format if needed
        format1 = workbook.add_format({'num_format': '#,##0.00'})
        worksheet.set_column('A:D', 18, format1)  # Example: Set format for columns A to D

# Save the DataFrame to a CSV file
# df.to_csv('exercise_results.csv', index=False)

# Confirming the DataFrame is saved and providing user feedback
print("Data has been saved to 'grouped_exercise_data.xlsx'.")

def update_maximums(player, exercise, actual_weight, actual_reps, e, r, s, w):
    """
    Update the player's maximum based on the actual performance versus the planned performance.
    
    :param player: Player's name as a string
    :param exercise: Exercise name as a string
    :param actual_weight: The weight actually lifted by the player
    :param actual_reps: The actual number of reps performed
    :param e: Exercise code used in percentage calculation
    :param r: Originally planned number of reps (not used here, provided for reference)
    :param s: Set number
    :param w: Week number
    """
    original_max = players_max_cores[player][exercise]
    
    # Get the assigned percentage based on the original plan
    assigned_percentage = P(e, r, s, w)  # Make sure this fetches the percentage correctly
    
    # Recalculate the actual percentage used based on actual performance
    # Assuming r is changed to actual_reps in the calculation
    actual_percentage = P(e, actual_reps, s, w)  # Recalculating with actual reps
    
    # Calculate the new maximum if actual_percentage is not zero to avoid division by zero
    if actual_percentage > 0:
        new_max = original_max * (assigned_percentage / actual_percentage)
        players_max_cores[player][exercise] = new_max
        print(f"Updated {player}'s {exercise} max from {original_max}lbs to {new_max}lbs based on performance.")
    else:
        print("Error: Actual percentage calculation resulted in zero or negative value, no update made.")

def recalculate_weights_for_player(player):
    updated_data = []
    for exercise, e, r, s, w in exercise_data:
        weight_output = W(player, e, r, s, w)  # Ensure W() fetches weights with updated max
        updated_data.append({
            "Player": player,
            "Exercise": exercise,
            "Week": w,
            "Set": s,
            "Reps": r,
            "Weight [lb]": weight_output
        })
    # Create DataFrame and sort by Week first, then by Exercise
    df = pd.DataFrame(updated_data)
    df = df.sort_values(by=['Week', 'Exercise'])
    return df

# Example: Suppose Johnson actually lifted a different weight than planned
update_maximums('Johnson', 'Squat', 390, 6, 1, 5, 1, 1)  # actual weight, actual reps, e, r, s, w

# Now, recalculate his weights with the updated maximum
updated_weights = recalculate_weights_for_player('Johnson')
#print(updated_weights)

# Assuming you have a dictionary players_max_cores and a list of exercises exercise_data
def update_and_save_weights():
    excel_path = 'updated_grouped_exercise_data.xlsx'
    
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        for player in players_max_cores.keys():
            # Recalculate and retrieve the sorted DataFrame for the player
            df = recalculate_weights_for_player(player)
            
            # Write each player's updated weights to a separate sheet
            df.to_excel(writer, sheet_name=player, index=False)
            
            # Excel formatting to improve readability
            worksheet = writer.sheets[player]
            worksheet.autofilter('A1:F1')  # Enable filtering for headers
            worksheet.set_column('A:B', 20)  # Set column width for Exercise names
            worksheet.set_column('F:F', 12)  # Set column width for Weight

    print(f"Updated weights saved to '{excel_path}'.")
# Execute the update and save process
update_and_save_weights()
    
### Define your expected results array in decimal form
##expected_P = [
##    0.65, 0.70, 0.80, 0.675, 0.725, 0.825, 0.70, 0.75, 0.85,
##    0.70, 0.77, 0.847, 0.6825, 0.735, 0.84, 0.7087, 0.7612, 0.8662,
##    0.735, 0.7875, 0.8925, 0.25, 0.25, 0.25, 0.65, 0.70, 0.80, 0.675,
##    0.725, 0.825, 0.70, 0.75, 0.85, 0.4, 0.4, 0.4, 0.325, 0.35, 0.4,
##    0.3375, 0.3625, 0.4125, 0.35, 0.375, 0.425, 0.1375, 0.1375, 0.1375,
##    0.65, 0.70, 0.875, 0.675, 0.725, 0.9, 0.70, 0.75, 0.925, 0.4062,
##    0.4375, 0.5, 0.4218, 0.4531, 0.5156, 0.4375, 0.4875, 0.5312, 0.075,
##    0.075, 0.075, 0.125, 0.125, 0.125, 0.65, 0.70, 0.80, 0.675, 0.725,
##    0.825, 0.70, 0.75, 0.85, 0.125, 0.125, 0.125
##]

### List of tuples, each representing (e, r, s, w)
##assignedERSW = [
##    (1, 5, 1, 1), (1, 5, 2, 1), (1, 10, 3, 1), (1, 3, 1, 2),
##    (1, 3, 2, 2), (1, 8, 3, 2), (1, 5, 1, 3), (1, 3, 2, 3),
##    (1, 7, 3, 3), (4, 5, 1, 1), (4, 4, 1, 2), (4, 3, 1, 3),
##    (6, 5, 1, 1), (6, 5, 2, 1), (6, 10, 3, 1), (6, 3, 1, 2),
##    (6, 3, 2, 2), (6, 8, 3, 2), (6, 5, 1, 3), (6, 3, 2, 3),
##    (6, 7, 3, 3), (12, 5, 1, 1), (12, 6, 1, 2), (12, 7, 1, 3),
##    (7, 5, 1, 1), (7, 5, 2, 1), (7, 10, 3, 1), (7, 3, 1, 2),
##    (7, 3, 2, 2), (7, 8, 3, 2), (7, 5, 1, 3), (7, 3, 2, 3),
##    (7, 7, 3, 3), (13, 3, 1, 1), (13, 4, 1, 2), (13, 5, 1, 3),
##    (2, 5, 1, 1), (2, 5, 2, 1), (2, 10, 3, 1), (2, 3, 1, 2),
##    (2, 3, 2, 2), (2, 8, 3, 2), (2, 5, 1, 3), (2, 3, 2, 3),
##    (2, 7, 3, 3), (8, 5, 1, 1), (8, 6, 1, 2), (8, 7, 1, 3),
##    (5, 5, 1, 1), (5, 5, 2, 1), (5, 6, 3, 1), (5, 3, 1, 2),
##    (5, 3, 2, 2), (5, 4, 3, 2), (5, 5, 1, 3), (5, 3, 2, 3),
##    (5, 3, 3, 3), (9, 5, 1, 1), (9, 5, 2, 1), (9, 10, 3, 1),
##    (9, 3, 1, 2), (9, 3, 2, 2), (9, 8, 3, 2), (9, 5, 1, 3),
##    (9, 3, 2, 3), (9, 7, 3, 3), (3, 3, 1, 1), (3, 4, 1, 2),
##    (3, 5, 1, 3), (10, 5, 1, 1), (10, 6, 1, 2), (10, 7, 1, 3),
##    (14, 5, 1, 1), (14, 5, 2, 1), (14, 10, 3, 1), (14, 3, 1, 2),
##    (14, 3, 2, 2), (14, 8, 3, 2), (14, 5, 1, 3), (14, 3, 2, 3),
##    (14, 7, 3, 3), (11, 7, 1, 1), (11, 8, 1, 2), (11, 10, 1, 3)
##]
##
### Check if the length of data and expected results match
##if len(assignedERSW) != len(expected_P):
##    print("Error: The number of data entries and expected results do not match.")
##else:
##    differences = []  # List to store the absolute differences
##    # Iterate over each set of parameters and compare the results
##    for parameters, expected in zip(assignedERSW, expected_P):
##        e, r, s, w = parameters
##        result = P(e, r, s, w)
##        difference = result - expected  # Calculate the difference
##        # Store the absolute value of the difference
##        differences.append(abs(difference))
##        print(f"p({e}, {r}, {s}, {w}) = {result:.4f}, Expected = {expected:.4f}, Difference = {difference:.4f}")
##    
##    # After the loop, find and display the maximum and minimum absolute differences
##    if differences:  # Ensure the list is not empty
##        max_difference = max(differences)
##        min_difference = min(differences)
##        print(f"Maximum absolute difference: {max_difference:.4f}")
##        print(f"Minimum absolute difference: {min_difference:.4f}")
##
