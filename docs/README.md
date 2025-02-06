# Lifting Program Automation

![Build Status](https://img.shields.io/github/workflow/status/rickystaugustine/NTlifting/CI?style=flat-square)  
![License](https://img.shields.io/github/license/rickystaugustine/NTlifting?style=flat-square)

## 📌 Overview
This project automates the process of assigning lifting weights for a high school football strength program. It integrates data from Google Sheets, calculates assigned weights based on tested maxes and multipliers, and simulates missing data for a conservative Functional Max Strength estimate.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Credentials Setup](#credentials-setup)
- [Workflow](#workflow)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [FAQ & Troubleshooting](#faq--troubleshooting)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Fixes & Updates](#fixes--updates)
- [How to Run Tests](#how-to-run-tests)
- [Maintainer](#maintainer)

## 🚀 Features
- **Google Sheets Integration**: Reads data from `CompleteProgram` and `Maxes` tabs.
- **Dynamic Weight Assignments**: Calculates weights using fitted multipliers.
- **Confidence-Based Simulations**: Estimates Functional Max Strength conservatively.
- **Data Dashboard**: Provides a structured summary for training analysis.
- **Error Handling**: Ensures robustness when handling missing or invalid data.

## 🔧 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/rickystaugustine/NTlifting.git
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. [Set up your credentials](#credentials-setup) for accessing Google Sheets.

## 📜 Credentials Setup
1. **Generate the credentials file:**
   - Go to the Google Cloud Console.
   - Enable the Google Sheets API.
   - Create a new service account and download the JSON credentials file.

2. **Store credentials securely:**
   - Place the JSON file in a secure location, such as `~/.config/your_project/credentials.json`.
   - Ensure that `credentials.json` is listed in `.gitignore` to prevent it from being committed to version control.

3. **Reference the credentials in your code:**
   - Set an environment variable, e.g., `GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json`.
   - Adjust your project’s configuration to load credentials from this environment variable or file path.

## 📊 Workflow
1. **Load Data** – Pulls lifting data from Google Sheets.
2. **Merge Data** – Aligns `CompleteProgram` and `Maxes` for calculations.
3. **Fit Multipliers** – Generates weight assignment formulas per exercise.
4. **Calculate Weights** – Computes and formats assigned weights.
5. **Simulate & Estimate Maxes** – Uses a conservative model for missing maxes.
6. **Generate Dashboard** – Summarizes key outputs for analysis.

## 🏗 Project Structure
### Consolidated Directory Descriptions:
- **`execution/`** → Core scripts for processing lifting programs and handling data.
- **`docs/`** → Documentation, including guides and references.
- **`tests/`** → Unit tests and other testing scripts.
- **`archive/`** → Deprecated or experimental files no longer in active use.

## 🔄 Usage
Run the main script:
```bash
python execution/main.py
```

## 📜 Scripts
- **`scripts/fetch_github_project.py`**: Retrieves GitHub project data and kanban boards for reporting purposes.

## 📜 Scripts
- **`scripts/fetch_github_project.py`**: Retrieves GitHub project data and kanban boards for reporting purposes.

## 📜 FAQ & Troubleshooting
- **Q:** How do I handle authentication errors with Google Sheets?  
  **A:** Make sure the credentials file is in the correct directory and that the service account has access to the target spreadsheet.

- **Q:** What do I do if there are missing maxes?  
  **A:** The script assigns "NRM" (No Relevant Max) and calculates weights accordingly. Check your source data to ensure it’s complete.

- **Q:** Why is the script failing at the merge step?  
  **A:** Verify that the `CompleteProgram` and `Maxes` tabs have consistent column names and that the data types match.

## 📜 CONTRIBUTING.md Location
Place the `CONTRIBUTING.md` file in the `docs/` directory to keep it easily accessible alongside other project documentation.

## 📌 Pull Request Guidelines
All pull requests should follow this naming format:
```
[TYPE] #ISSUE_NUMBER - Short Description
```
### **Examples:**
```
[BUGFIX] #7 - Fix Google Sheets Data Loading
[FEATURE] #15 - Implement Weight Progression Model
[DOCS] #22 - Update README Formatting
```
### **PR Types:**
- `[FEATURE]` → New feature implementation
- `[BUGFIX]` → Fixes a bug
- `[REFACTOR]` → Code cleanup, optimizations
- `[DOCS]` → Documentation updates
- `[TEST]` → Unit tests, test automation
- `[CONFIG]` → CI/CD or configuration changes

### References
- **Pull Request Template:** Found at `.github/PULL_REQUEST_TEMPLATE.md`.
- **GitHub Workflow - Report Generation:** See `.github/workflows/github_report.yml`.
- **GitHub Workflow - Test Suite:** See `.github/workflows/test-suite.yml`.

## 🔧 Fixes & Updates

### ✅ Full Test Suite Passing (Latest Fixes)
- **Resolved `KeyError: 'Tested Max'`** in `test_assign_weights.py` by ensuring `"Tested Max"` is included in test data.
- **Fixed dtype compatibility issue** with `"NRM"` assignments in Pandas.  
  - Explicitly converted `"Assigned Weight"` to `object` before assigning `"NRM"`.
- **Corrected `write_to_google_sheet()` behavior** to properly append rows even when creating a new worksheet.
- **Updated `test_google_sheets_utils.py` mocking** to properly simulate worksheet creation and prevent false failures.
- **Ensured `fit_multipliers()` accepts DataFrame input** instead of incorrectly receiving a dictionary.
- **Fixed `run_simulation()` function signature** to accept expected input arguments.

---

### ✅ How to Run Tests
To verify functionality, run:
```bash
pytest tests/ --disable-warnings
```

## 🏆 Maintainer
**Ricky St. Augustine**  
For questions, reach out via GitHub Issues.

