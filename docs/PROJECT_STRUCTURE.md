# NTlifting Project Structure

This document provides an overview of the NTlifting project structure, detailing the purpose of each directory and key files.

---

## **📂 Root Directory (`NTlifting/`):**
The main project directory containing execution scripts, configurations, and documentation.

- **`execution/`** → Core processing scripts & helpers.
- **`docs/`** → Project documentation.
- **`outputs/`** → Generated reports, plots, and analysis results.
- **`archive/`** → Old, deprecated, or experimental files.
- **`project_structure.txt`** → The most up-to-date project structure tracking file.
- **`.gitignore`** → Git version control exclusions.
- **`README.md`** → Overview of the project.
- **`requirements.txt`** → Dependencies required for the project.

---

## **📂 Execution Directory (`execution/`):**
Contains core scripts, organized by functionality.

- **`main.py`** → Entry point for processing the lifting program.
- **`evaluation/`** → Scripts for evaluating the model’s accuracy & effectiveness.
- **`analysis/`** → Scripts for reviewing results & validating calculations.
- **`helpers/`** → Utility modules for data loading, processing, and calculations.
- **`inputs/`** → Scripts interacting with Google Sheets & other input sources.
- **`experimental/`** → Work-in-progress or test scripts.

---

## **📂 Evaluation Directory (`execution/evaluation/`):**
Handles model evaluation, error analysis, and confidence calculations.

- **`functional_max_review.py`** → Reviews the accuracy of Functional Max calculations.
- **`functional_max_summary_metrics.csv`** → Summary of Functional Max results.
- **`functional_max_diff_histogram.png`** → Histogram visualization of Functional Max deviations.

---

## **📂 Analysis Directory (`execution/analysis/`):**
Handles reviewing and validating different stages of the process.

- **`assigned_weights_review.py`** → Analyzes assigned weights.
- **`fitting_review.py`** → Evaluates how well multipliers fit the data.
- **`simulated_data_review.py`** → Assesses simulated lifting results.
- **`functional_max_review.py`** → Validates Functional Max calculations.
- **`functional_max_validation.py`** → Ensures Functional Max estimates align with expectations.

### **Subdirectories:**
- **`assigned_weights_review/`** → Contains detailed plots & CSV files of assigned weight evaluations.
- **`fitting_review/`** → Stores error metrics & multiplier fit results.
- **`functional_max_review/`** → Functional Max result distributions & summary stats.
- **`simulated_data_review/`** → Review and analysis of simulated lifting data.

---

## **📂 Helpers Directory (`execution/helpers/`):**
Utility scripts supporting core processing.

- **`data_loading.py`** → Loads data from Google Sheets.
- **`data_processing.py`** → Prepares and cleans input data.
- **`multiplier_fitting.py`** → Fits multipliers to exercises.
- **`weight_assignment.py`** → Assigns appropriate lifting weights.
- **`simulation.py`** → Simulates lifting data.
- **`functional_max_calc.py`** → Calculates Functional Max based on simulations.
- **`google_sheets_utils.py`** → Handles reading/writing to Google Sheets.

---

## **📂 Outputs Directory (`outputs/`):**
Stores generated results, plots, and analysis outputs.

- **`Plots/`** → Various visualizations of lifting performance.
  - `rep_differences.png` → Differences between assigned and actual reps.
  - `functional_tested_differentials.png` → Comparison of Functional vs. Tested Max.
  - `weight_differences/` → Weight assignment accuracy and analysis.
- **`metrics_history.csv`** → Tracks error metrics over time.
- **`WeightDifferencePlots/`** → Moved to `execution/analysis/` for weight difference analysis.

---

## **📂 Archive Directory (`archive/`):**
Stores old, deprecated, or experimental data and scripts.

- **`documentation/`** → Old project structure files & docs.
- **`deprecated_scripts/`** → Scripts no longer in use.
- **`logs/`** → Historical logs and debugging files.
- **`reference_models/`** → Legacy models for comparison.
- **`experimental/`** → In-progress or conceptual scripts.
- **`images/`** → Previously generated plots and analysis images.

---

## **📂 Documentation (`docs/`):**
Contains user-friendly documentation and references.

- **`PROJECT_STRUCTURE.md`** → This document.
- **`updated_project_structure.txt`** → Previously tracked structure (archived).
- **`README.md`** → Project introduction and instructions.

---

## **Next Steps:**
✅ **Update this document** whenever the project structure changes.  
✅ **Maintain `project_structure.txt`** in the root directory for future tracking.  
✅ **Ensure `docs/PROJECT_STRUCTURE.md` remains readable & accessible.**

