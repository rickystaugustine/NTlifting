# **NT Lifting Performance Model**

## **Overview**
This project models and analyzes **lifting performance and Functional Max calculations** for athletes. It integrates **Google Sheets** to manage data, fits **exercise-specific multipliers**, generates **customized training programs**, and simulates **realistic weight execution** based on statistical models. It also **tracks error convergence** over multiple iterations and validates execution results for accuracy.

## **Project Structure**
```
NTlifting/
│── core_scripts/
│   │── exercise_performance_model.py  # Main execution script
│   │── fitting_and_simulation.py      # Fits exercise models, simulates data
│   │── google_sheets.py               # Handles Google Sheets interaction
│   │── visualization.py               # Generates plots for Functional Max analysis
│── experimental/
│   │── IncorporateActualData.py       # Uses executional data for Functional Max calculations
│   │── FunctionalMaxConfidence.py     # Evaluates confidence in Functional Max based on simulation
│── archive/                           # Stores deprecated/reference scripts
│── configs/                           # Contains authentication files (not in GitHub)
│── docs/                              # Documentation
│── logs/                              # Execution logs & model accuracy tracking
│── venv_NTlifting/                    # Virtual environment (ignored by Git)
│── run_pipeline.py                    # Orchestrates execution workflow
│── .gitignore
│── requirements.txt                    # Dependencies list
```

## **Core Scripts**
### **1️⃣ Main Execution: `exercise_performance_model.py`**
- Reads **training data** from Google Sheets.
- Fits **exercise multipliers** to model lifting progression.
- Generates **personalized lifting programs**.
- Simulates **lift execution** based on probability distributions.
- Computes **Functional Max strength based on actual vs. assigned reps & weights**.
- Analyzes **execution accuracy & error convergence**.
- **Validates execution results to ensure proper model accuracy.**
- Writes updated **results back to Google Sheets**.

### **2️⃣ Multipliers & Simulation: `fitting_and_simulation.py`**
- Defines **multiplier function equations**.
- Classifies **constant & fitted multipliers**.
- Fits models for **each exercise**.
- Generates **actual rep differences** based on probability distributions.
- Runs **chunk-based lift execution simulations**.
- **Implements probability-based rep execution modeling to ensure realistic performance simulation.**

### **3️⃣ Google Sheets Integration: `google_sheets.py`**
- **Authenticates** using service account credentials.
- Handles **structured data retrieval and updates for tracking lifting progress**.
- Reads structured data from **Google Sheets**.
- Writes processed results **back to Google Sheets**.

### **4️⃣ Data Visualization: `visualization.py`**
- **Plots Functional Max differences** to analyze distribution.
- Generates **performance insights** based on rep & weight variations.
- Saves **local PNG outputs** for further review.

## **Experimental Scripts**
These scripts are not actively used but remain for reference and future improvements:

### **1️⃣ `IncorporateActualData.py`**
- **Initial attempt** at integrating executional data to calculate Functional Max strength.

### **2️⃣ `FunctionalMaxConfidence.py`**
- **Preliminary model** to evaluate confidence in Functional Max calculations based on simulation data.

## **Execution Workflow**
Instead of running scripts manually, use **`run_pipeline.py`** to orchestrate everything:
```bash
python3 run_pipeline.py
```
This will:
1. **Preprocess and load training data** from Google Sheets.
2. **Fit models for exercise multipliers**.
3. **Generate lifting programs & simulate execution**.
4. **Analyze & visualize Functional Max performance**.
5. **Validate execution results to ensure modeling accuracy.**
6. **Write results back to Google Sheets** for tracking.

## **Configuration & Authentication**
Certain authentication files for Google Sheets **are stored locally** and are **not committed to GitHub**:
```
configs/
    ├── client_secret.json
    ├── ntafterschoollifting-b8f7a5923646.json
```
Ensure these files are correctly placed in **`/Users/ricky.staugustine/Documents/FB/Lifting/`**.

## **Dependencies & Setup**
Install required dependencies using:
```bash
pip install -r requirements.txt
```

## **Future Development**
- Optimize **Functional Max confidence analysis**.
- Refine **incorporation of executional data**.
- Improve **error convergence tracking and execution validation metrics**.
- Expand **statistical modeling of lift execution distributions**.
- **Develop additional simulation techniques to refine lifting program accuracy.**
- **Enhance performance optimizations for large-scale data processing.**

## **Contributors**
- **Ricky St. Augustine** (Project Owner & Developer)

---
📌 **For any issues or improvements, update this README and track changes in GitHub.** 🚀

