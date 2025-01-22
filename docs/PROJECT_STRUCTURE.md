# 📁 NTlifting Project Structure

## 🏷️ Overview
This project is designed for **analyzing and modeling strength training progression**, specifically for **functional max calculations** based on training data.

### **📂 Root Directories**
| Directory                  | Purpose |
|---------------------------|---------|
| `execution/`              | Core execution scripts for analysis, experimental testing, and helpers. |
| `docs/`                   | Documentation, including README, requirements, and structure notes. |
| `config/`                 | Configuration files (e.g., API credentials, user settings). |
| `archive/`                | Deprecated files, logs, and previous iterations of scripts. |
| `venv_NTlifting/`         | Virtual environment containing dependencies. |

---

### **📂 `execution/` (Main Project Code)**

| Directory                    | Purpose |
|-----------------------------|---------|
| `execution/analysis/`       | Review scripts for validating calculations, analyzing data, and ensuring accuracy. |
| `execution/experimental/`   | Scripts for testing new models before integrating them into the main pipeline. |
| `execution/helpers/`        | Utility functions for multipliers, Google Sheets interaction, and simulations. |
| `execution/outputs/`        | Stores CSV results, plots, and execution logs. |
| `execution/logs/`           | Stores execution logs for debugging and tracking. |

#### **Key Scripts**
- `exercise_performance_model.py` → **Main execution script** for functional max calculations.
- `run_analysis.sh` → **Automated script** for running all analyses sequentially.

---

### **📂 `execution/analysis/` (Validation & Review Scripts)**

| Subdirectory                 | Purpose |
|-----------------------------|---------|
| `assigned_weights_review/`  | Ensures assigned weights match training expectations. |
| `fitting_review/`           | Validates the accuracy of multiplier function fitting. |
| `functional_max_review/`    | Analyzes functional max calculation accuracy. |
| `simulated_data_review/`    | Checks correctness of simulated training results. |

---

### **📂 `execution/experimental/` (Testing & Research Scripts)**

| Script Name                         | Purpose |
|-------------------------------------|---------|
| `calculate_functional_max.py`       | Experimental script for computing functional max. |
| `FunctionalMaxConfidence.py`        | Evaluates confidence intervals in functional max estimations. |
| `IncorporateActualData.py`          | Prototype for integrating real training data. |

---

### **📂 `execution/helpers/` (Utility Functions & APIs)**

| Script Name                     | Purpose |
|---------------------------------|---------|
| `fitting.py`                    | Handles multiplier function fitting. |
| `multipliers.py`                 | Defines models for exercise multipliers. |
| `google_sheets_utils.py`         | Reads/writes training data from Google Sheets. |
| `simulation.py`                  | Generates simulated training sets. |

---

### **📂 `execution/outputs/` (Stored Results & Plots)**

| Subdirectory                | Purpose |
|----------------------------|---------|
| `Data/`                    | Stores CSV files of analysis results. |
| `Plots/`                   | Stores images generated from model evaluations. |
| `logs/`                    | Execution logs for debugging. |

---

## **📌 Expected Input & Output Formats**
### **Input Data Sources**
- **Google Sheets:** Training data is pulled from `After-School Lifting` Google Sheet.
- **CSV Files:** Processed data is stored and reloaded from `execution/outputs/Data/`.

### **Key Input Columns**
| Column Name        | Description |
|-------------------|-------------|
| `Player`         | Athlete name |
| `Exercise`       | Name of the exercise performed |
| `Week #`         | Training week |
| `Set #`          | Set number within a workout |
| `# of Reps`      | Assigned repetitions |
| `Multiplier`     | Assigned weight multiplier |
| `Tested Max`     | Athlete's last known max strength |

### **Key Output Columns**
| Column Name        | Description |
|-------------------|-------------|
| `Assigned Weight` | Computed weight to be lifted |
| `Actual Reps`     | Simulated reps performed |
| `Actual Weight`   | Simulated weight lifted |
| `Functional Max`  | Estimated strength based on model |

---

## **📌 How to Run Everything**
To execute the full analysis pipeline, run:
```bash
./run_analysis.sh
```
This will:
1. Compute Functional Max.
2. Review Functional Max accuracy.
3. Validate assigned weight calculations.
4. Analyze multiplier fitting accuracy.

---

## **📌 Next Steps**
- Integrate real training data.
- Optimize functional max calculation for improved accuracy.
- Expand confidence interval evaluations.

🚀 **This document will be updated as the project evolves!**


