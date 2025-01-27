import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../execution")))

def review_functional_max(output_dir="execution/analysis/functional_max_review"):
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Starting review of Functional Max calculations...")
    
    # Ensure simulated data review file exists before proceeding
    review_file_path = os.path.join(output_dir, "functional_max_results.csv")
    if not os.path.exists(review_file_path):
        logging.error(f"Missing file: {review_file_path}. Ensure that Functional Max calculations have been generated before running this script.")
        return
    
    # Load Functional Max data from the review file
    functional_max_df = pd.read_csv(review_file_path)
    
    # Compute differences between Functional Max and Tested Max
    functional_max_df["FM - TM"] = functional_max_df["Functional Max"] - functional_max_df["Tested Max"]
    
    # Save review data
    functional_max_df.to_csv(os.path.join(output_dir, "functional_max_review.csv"), index=False)
    logging.info(f"Functional Max review data saved to {output_dir}/functional_max_review.csv")
    
    # Generate distribution plots
    plt.figure(figsize=(12, 6), dpi=150)
    num_bins = min(50, max(10, len(functional_max_df) // 10))  # Dynamically adjust bins based on data size
    plt.hist(functional_max_df["FM - TM"], bins=num_bins, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel("Functional Max - Tested Max")
    plt.ylabel("Frequency")
    plt.title("Distribution of Functional Max Differences")
    plt.savefig(os.path.join(output_dir, "functional_max_distribution.png"))
    plt.close()
    
    # KDE Plot
    plt.figure(figsize=(12, 6), dpi=150)
    functional_max_df["FM - TM"].plot(kind='kde', color='red')
    plt.xlabel("Functional Max - Tested Max")
    plt.ylabel("Density")
    plt.title("KDE Plot of Functional Max Differences")
    plt.savefig(os.path.join(output_dir, "functional_max_kde.png"))
    plt.close()
    
    # Scatter plot: Functional Max vs Tested Max
    plt.figure(figsize=(12, 8), dpi=150)
    jitter = np.random.uniform(-0.5, 0.5, size=len(functional_max_df))
    plt.scatter(functional_max_df["Tested Max"] + jitter, functional_max_df["Functional Max"] + jitter, alpha=0.6, s=150, edgecolors='black')
    plt.plot(functional_max_df["Tested Max"], functional_max_df["Tested Max"], linestyle='dashed', color='black', label="Tested Max Line")
    plt.xlabel("Tested Max")
    plt.ylabel("Functional Max")
    plt.title("Functional Max vs Tested Max")
    plt.legend()
    scatter_plot_path = os.path.join(output_dir, "functional_max_vs_tested_max.png")
    plt.savefig(scatter_plot_path)
    plt.close()
    
    # Log statistics
    mean_diff = functional_max_df["FM - TM"].mean()
    median_diff = functional_max_df["FM - TM"].median()
    std_dev = functional_max_df["FM - TM"].std()
    skewness = functional_max_df["FM - TM"].skew()
    
    logging.info(f"Mean Functional Max Difference: {mean_diff}")
    logging.info(f"Median Functional Max Difference: {median_diff}")
    logging.info(f"Standard Deviation: {std_dev}")
    logging.info(f"Skewness: {skewness}")
    
    if os.path.exists(scatter_plot_path):
        logging.info(f"Functional Max review plots successfully saved to {output_dir}/")
    else:
        logging.error(f"Failed to save Functional Max review plots to {output_dir}/")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    review_functional_max()
