import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

def generate_aggregate_plots(expanded_df):
    """
    Generate aggregate plots for evaluating Functional Max deviations and trend smoothing.
    """
    # Suppress Matplotlib INFO logs
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    output_dir = "plots/aggregates"
    os.makedirs(output_dir, exist_ok=True)

    # Functional Max Deviation from Tested Max (per Exercise)
    expanded_df['FM Deviation'] = expanded_df['Functional Max'] - expanded_df['Tested Max']

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Code', y='FM Deviation', data=expanded_df)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Functional Max Deviation from Tested Max per Exercise')
    plt.xlabel('Exercise Code')
    plt.ylabel('Functional Max Deviation (lbs)')
    plt.savefig(os.path.join(output_dir, 'fm_deviation_per_exercise.png'))
    plt.close()

    # Trend Estimate Deviation from Functional Max (SMA example)
    if 'SMA' in expanded_df.columns:
        expanded_df['SMA Deviation'] = expanded_df['SMA'] - expanded_df['Functional Max']

        plt.figure(figsize=(12, 6))
        sns.lineplot(
            x='Week #',
            y='SMA Deviation',
            hue=expanded_df['Code'].astype(str),
            data=expanded_df
        )
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title('SMA Trend Estimate Deviation from Functional Max over Weeks')
        plt.xlabel('Week')
        plt.ylabel('Deviation (SMA - Functional Max)')
        plt.savefig(os.path.join(output_dir, 'sma_deviation_over_weeks.png'))
        plt.close()

    logging.info("✅ Aggregate plots generated and saved to 'plots/aggregates/'")
