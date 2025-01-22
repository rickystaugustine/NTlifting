import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../execution")))

print("PYTHON SEARCH PATHS:")
for p in sys.path:
    print(p)
from helpers.google_sheets_utils import read_google_sheets

def review_assigned_weights(output_dir="execution/analysis/assigned_weights_review"):
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Starting review of assigned weights...")
    
    # Load data from Google Sheets
    assigned_weights_df = read_google_sheets("After-School Lifting", "AssignedWeights")
    
    # Ensure necessary columns exist
    required_columns = ["Player", "Exercise", "Relevant Core", "Tested Max", "Assigned Weight"]
    if not all(col in assigned_weights_df.columns for col in required_columns):
        logging.error("Missing required columns in AssignedWeights data.")
        return
    
    # Compute percentage difference
    assigned_weights_df["Percentage Difference"] = (
        (assigned_weights_df["Tested Max"] - assigned_weights_df["Assigned Weight"]) / assigned_weights_df["Tested Max"] * 100
    )
    
    # Save the data for review
    assigned_weights_df.to_csv(os.path.join(output_dir, "assigned_weights_review.csv"), index=False)
    logging.info(f"Assigned weights review saved to {output_dir}/assigned_weights_review.csv")
    
    # Generate plots
    players = assigned_weights_df["Player"].unique()
    for player in players:
        player_data = assigned_weights_df[assigned_weights_df["Player"] == player]
        exercises = player_data["Exercise"].unique()
        plt.figure(figsize=(12, 8), dpi=150)
        
        # Use a more distinctive colormap
        colors = plt.colormaps["tab20"].colors  # Better distinction for categories
        color_map = {exercise: colors[i % len(colors)] for i, exercise in enumerate(exercises)}
        markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', 'H', '<', '>', 'p']  # Different marker styles
        
        for idx, exercise in enumerate(exercises):
            exercise_data = player_data[player_data["Exercise"] == exercise]
            plt.scatter(exercise_data["Tested Max"], exercise_data["Assigned Weight"], 
                        label=exercise, alpha=0.8, s=150, 
                        color=color_map[exercise], marker=markers[idx % len(markers)], edgecolors='black')
        
        plt.plot(player_data["Tested Max"], player_data["Tested Max"], linestyle='dashed', color='black', label="Tested Max")
        plt.xlabel("Tested Max")
        plt.ylabel("Assigned Weight")
        plt.title(f"Assigned Weights vs. Tested Max - {player}")
        plt.legend(title="Exercise", loc='best', fontsize=8, markerscale=1.5)
        plt.savefig(os.path.join(output_dir, f"assigned_weights_{player}.png"))
        plt.close()
    logging.info(f"Assigned weights plots saved to {output_dir}/")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    review_assigned_weights()
