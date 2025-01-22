import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
import os

# Ensure Python can locate `helpers/`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from helpers.fitting import fit_single_exercise_global
from helpers.multipliers import ConstantMultiplier, FittedMultiplier

def review_fitting_analysis(program_df, output_dir="execution/analysis/fitting_review"):
    metrics_dir = os.path.join(output_dir, "metrics")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    logging.info("Starting detailed review of exercise fitting process...")
    
    exercises = program_df['Code'].unique()
    program_records = program_df.to_dict("records")
    exercise_functions = {}
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "fitting_equations.txt"), "w") as eq_file:
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(fit_single_exercise_global, int(code), program_records): int(code) for code in exercises}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    func_tuple = future.result()
                    multiplier = func_tuple[1]
                    
                    # Ensure multipliers are properly classified
                    unique_multipliers = program_df[program_df['Code'] == code]['Multiplier of Max'].unique()
                    if len(unique_multipliers) > 1:
                        multiplier = FittedMultiplier(func_tuple[1].params) if isinstance(multiplier, FittedMultiplier) else FittedMultiplier([0, 0, 0, np.mean(unique_multipliers)])
                    
                    exercise_functions[int(code)] = (code, multiplier)
                    
                    if isinstance(multiplier, FittedMultiplier):
                        equation = f"{multiplier.params[0]} * w + {multiplier.params[1]} * s + {multiplier.params[2]} * np.log(r + 1) + {multiplier.params[3]}"
                        eq_file.write(f"Exercise {code}: {equation}\n")
                    else:
                        eq_file.write(f"Exercise {code}: Constant Multiplier {multiplier.value}\n")
                    
                    logging.info(f"Reviewed fit for exercise code {code}.")
                except Exception as e:
                    logging.error(f"Error processing exercise code {code}: {e}")
    
    fit_results = pd.DataFrame.from_dict(
        {k: (type(v[1]).__name__, v[1].params if isinstance(v[1], FittedMultiplier) else v[1].value) for k, v in exercise_functions.items()},
        orient='index', columns=['Multiplier Type', 'Parameters']
    )
    fit_results.to_csv(os.path.join(metrics_dir, "fitting_results.csv"))
    
    cached_multipliers = {code: program_df[program_df['Code'] == code] for code in exercises}
    
    error_metrics = []
    for code, (_, multiplier) in exercise_functions.items():
        exercise_data = cached_multipliers[code]
        assigned_multipliers = exercise_data['Multiplier of Max'].values
        weeks = exercise_data['Week #'].values
        sets = exercise_data['Set #'].values
        reps = exercise_data['# of Reps'].values
        calculated_multipliers = [multiplier(w, s, r) for w, s, r in zip(weeks, sets, reps)]
        
        if len(calculated_multipliers) > 0 and len(assigned_multipliers) > 0:
            mae = np.mean(np.abs(np.array(calculated_multipliers) - np.array(assigned_multipliers)))
            rmse = np.sqrt(np.mean((np.array(calculated_multipliers) - np.array(assigned_multipliers))**2))
            error_metrics.append([code, mae, rmse])
        
        plt.figure()
        if len(weeks) == len(assigned_multipliers):
            plt.scatter(weeks, assigned_multipliers, label='Assigned Multipliers', color='blue')
        else:
            logging.warning(f"Skipping plot for Exercise {code} due to mismatched data lengths: weeks({len(weeks)}) vs assigned_multipliers({len(assigned_multipliers)})")
        plt.scatter(weeks, calculated_multipliers, label='Fitted Multipliers', color='red', alpha=0.7)
        plt.xlabel('Weeks')
        plt.ylabel('Multiplier')
        plt.title(f'Assigned vs. Fitted Multipliers for Exercise {code}')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f"exercise_{code}_comparison.png"))
        plt.close()
    
    error_metrics_df = pd.DataFrame(error_metrics, columns=["Exercise Code", "MAE", "RMSE"])
    error_metrics_df.to_csv(os.path.join(metrics_dir, "fitting_error_metrics.csv"), index=False)
    logging.info(f"Error metrics saved to {metrics_dir}/fitting_error_metrics.csv")
    
    for code in exercises:
        logging.info(f"Plot saved: {os.path.join(plots_dir, f'exercise_{code}_comparison.png')}")
    logging.info(f"Fitting plots saved to {plots_dir}/")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    from helpers.google_sheets_utils import read_google_sheets
    program_df = read_google_sheets("After-School Lifting", "CompleteProgram")
    review_fitting_analysis(program_df)
