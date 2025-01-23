import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define paths
EVAL_DIR = "execution/evaluation/functional_max/"
INPUT_FILE = "execution/analysis/functional_max_review/functional_max_review.csv"
OUTPUT_PLOT = os.path.join(EVAL_DIR, "functional_max_diff_histogram.png")
OUTPUT_METRICS = os.path.join(EVAL_DIR, "functional_max_summary_metrics.csv")

# Ensure directory exists
os.makedirs(EVAL_DIR, exist_ok=True)

# 📌 1. Load Functional Max Review Data
df = pd.read_csv(INPUT_FILE)

# Ensure the required columns exist
assert "FM - TM" in df.columns, "Missing column 'FM - TM' (Functional Max - Tested Max)"

# 📌 2. Compute Summary Statistics
mean_diff = df["FM - TM"].mean()
median_diff = df["FM - TM"].median()
std_dev = df["FM - TM"].std()
skewness = df["FM - TM"].skew()

# 📌 3. Adjust Confidence Calculation (Bias Toward Underestimation)
confidence_lower = np.percentile(df["FM - TM"], 25)  # 25th percentile
confidence_upper = np.percentile(df["FM - TM"], 75)  # 75th percentile
conservative_max = np.percentile(df["FM - TM"], 5)  # 5th percentile

# 📌 4. Save Summary Metrics
summary = pd.DataFrame({
    "Mean Difference": [mean_diff],
    "Median Difference": [median_diff],
    "Standard Deviation": [std_dev],
    "Skewness": [skewness],
    "25th Percentile (Conf. Lower)": [confidence_lower],
    "75th Percentile (Conf. Upper)": [confidence_upper],
    "5th Percentile (Conservative Max)": [conservative_max]
})
summary.to_csv(OUTPUT_METRICS, index=False)

# 📌 5. Plot Distribution of Functional Max Differences
plt.figure(figsize=(10, 6))
sns.histplot(df["FM - TM"], bins=40, kde=True, alpha=0.75)

# Add mean & median lines
plt.axvline(mean_diff, color='r', linestyle="dashed", linewidth=2, label="Mean")
plt.axvline(median_diff, color='b', linestyle="dashed", linewidth=2, label="Median")

# Titles & labels
plt.title("Distribution of Functional Max - Tested Max Differences")
plt.xlabel("Functional Max - Tested Max (lbs)")
plt.ylabel("Frequency")
plt.legend()

# Save Plot
plt.savefig(OUTPUT_PLOT)
plt.close()

print(f"✅ Functional Max Evaluation Complete! Results saved to:\n- {OUTPUT_METRICS}\n- {OUTPUT_PLOT}")

