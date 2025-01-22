import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../execution")))

from helpers.google_sheets_utils import read_google_sheets

def review_simulated_data(output_dir="execution/analysis/simulated_data_review"):
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Starting review of simulated lift data...")
    
    # Load simulated lift data from Google Sheets
    simulated_data_df = read_google_sheets("After-School Lifting", "SimulatedLiftData")
    
    # Ensure necessary columns exist
    required_columns = ["Player", "Exercise", "Tested Max", "Multiplier", "Assigned Weight", "Actual Reps", "Actual Weight", "# of Reps"]
    if not all(col in simulated_data_df.columns for col in required_columns):
        logging.error("Missing required columns in SimulatedLiftData sheet.")
        return
    
    # Compute differences
    simulated_data_df["Weight Difference"] = simulated_data_df["Actual Weight"] - simulated_data_df["Assigned Weight"]
    simulated_data_df["Reps Difference"] = simulated_data_df["Actual Reps"] - simulated_data_df["# of Reps"]
    
    # Save the data for review
    simulated_data_df.to_csv(os.path.join(output_dir, "simulated_data_review.csv"), index=False)
    logging.info(f"Simulated data review saved to {output_dir}/simulated_data_review.csv")
    
    # Generate weight comparison plots
    players = simulated_data_df["Player"].unique()
    for player in players:
        player_data = simulated_data_df[simulated_data_df["Player"] == player]
        exercises = player_data["Exercise"].unique()
        plt.figure(figsize=(12, 8), dpi=150)
        
        colors = plt.colormaps["tab20"].colors
        color_map = {exercise: colors[i % len(colors)] for i, exercise in enumerate(exercises)}
        markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', 'H', '<', '>', 'p']
        
        for idx, exercise in enumerate(exercises):
            exercise_data = player_data[player_data["Exercise"] == exercise]
            plt.scatter(exercise_data["Assigned Weight"], exercise_data["Actual Weight"], 
                        label=exercise, alpha=0.8, s=150, 
                        color=color_map[exercise], marker=markers[idx % len(markers)], edgecolors='black')
        
        plt.plot(player_data["Assigned Weight"], player_data["Assigned Weight"], linestyle='dashed', color='black', label="Assigned Weight")
        plt.xlabel("Assigned Weight")
        plt.ylabel("Actual Weight")
        plt.title(f"Assigned vs. Actual Weight - {player}")
        plt.legend(title="Exercise", loc='best', fontsize=8, markerscale=1.5)
        plt.savefig(os.path.join(output_dir, f"simulated_data_{player}_weight.png"))
        plt.close()
    
    # Generate reps comparison plots
    for player in players:
        player_data = simulated_data_df[simulated_data_df["Player"] == player]
        plt.figure(figsize=(12, 8), dpi=150)
        
        for idx, exercise in enumerate(exercises):
            exercise_data = player_data[player_data["Exercise"] == exercise]
            plt.scatter(exercise_data["# of Reps"], exercise_data["Actual Reps"], 
                        label=exercise, alpha=0.8, s=150, 
                        color=color_map[exercise], marker=markers[idx % len(markers)], edgecolors='black')
        
        plt.plot(player_data["# of Reps"], player_data["# of Reps"], linestyle='dashed', color='black', label="Assigned Reps")
        plt.xlabel("Assigned Reps")
        plt.ylabel("Actual Reps")
        plt.title(f"Assigned vs. Actual Reps - {player}")
        plt.legend(title="Exercise", loc='best', fontsize=8, markerscale=1.5)
        plt.savefig(os.path.join(output_dir, f"simulated_data_{player}_reps.png"))
        plt.close()
    logging.info(f"Simulated data plots saved to {output_dir}/")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    review_simulated_data()
