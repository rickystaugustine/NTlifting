# NT Lifting Project

## **Overview**
This project is designed to analyze and optimize weightlifting performance by modeling exercise performance, calculating functional max estimates, and simulating weight adjustments based on executed reps and weight differences.

## **Project Structure**
The project has been **reorganized** for clarity and efficiency. Below is the updated directory structure:

### **Active Scripts**
- `execution/exercise_performance_model.py` - **Main script** for running performance analysis.
- `models/fitting_and_simulation.py` - **Helper script** for model fitting and simulation functions.

### **Core Directories**
- `execution/` - Contains scripts that are actively run.
- `models/` - Contains helper functions for exercise fitting and simulation.
- `configs/` - Holds configuration files.
- `docs/` - Project documentation, including `README.md` and `requirements.txt`.
- `outputs/Plots/` - Stores generated analysis plots.
- `logs/` - Contains logs, including error tracking and historical metrics.
- `archive/` - Stores deprecated scripts and old project components for reference.

## **Inputs & Outputs**
### **Inputs**
- Google Sheets data accessed via `gspread`
- Configurations stored in `configs/`

### **Outputs**
- **Functional vs. Tested Max Differences:** `outputs/Plots/functional_tested_differentials.png`
- **Rep Differences:** `outputs/Plots/rep_differences.png`
- **Weight Difference Analysis:** `outputs/Plots/WeightDifferencePlots/`
- **Model Metrics:** `logs/metrics_history.csv`

## **Installation & Setup**
### **1. Create & Activate Virtual Environment**
```sh
python3 -m venv venv_NTlifting
source venv_NTlifting/bin/activate
```

### **2. Install Dependencies**
```sh
pip install -r docs/requirements.txt
```

### **3. Running the Script**
To execute the model:
```sh
python execution/exercise_performance_model.py
```

## **Future Work**
- Refactoring core components for modularization.
- Further improving simulation accuracy.
- Automating more aspects of weight prediction.

---
This `README.md` has been updated to reflect the **new project organization** as of the latest refactor.

