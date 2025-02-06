# # Lifting Program Automation

A streamlined solution for automating lifting weight assignments and simulations.

\


## Overview
The project is currently stable, with key features in place for reading and writing to Google Sheets. Most issues involve refining existing workflows rather than building new ones from scratch.

This project automates the process of assigning lifting weights for a high school football strength program. It integrates data from Google Sheets, calculates assigned weights based on tested maxes and multipliers, and simulates missing data for a conservative Functional Max Strength estimate.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Credentials Setup](#credentials-setup)
- [Workflow](#workflow)
- [Project Structure](#project-structure)
- [Key Project Inputs](#key-project-inputs)
- [Key Project Outputs](#key-project-outputs)
- [Usage](#usage)
- [FAQ & Troubleshooting](#faq--troubleshooting)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Fixes & Updates](#fixes--updates)
- [How to Run Tests](#how-to-run-tests)
- [Maintainer](#maintainer)

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Credentials Setup](#credentials-setup)
- [Workflow](#workflow)
- [Project Structure](#project-structure)
- [Key Project Inputs](#key-project-inputs)
- [Key Project Outputs](#key-project-outputs)
- [Usage](#usage)
- [FAQ & Troubleshooting](#faq--troubleshooting)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Fixes & Updates](#fixes--updates)
- [How to Run Tests](#how-to-run-tests)
- [Maintainer](#maintainer)

## Features

- **Google Sheets Integration:** Reads data from `CompleteProgram` and `Maxes` tabs. (Some complexity in handling different sheet structures.)
- **Dynamic Weight Assignments:** Calculates weights using fitted multipliers. (Complex due to reliance on properly merged data.)
- **Confidence-Based Simulations:** Estimates Functional Max Strength conservatively. (Prone to errors if the input data ranges are inconsistent.)
- **Functional Max Calculations:** Currently in development. (Complex logic still being refined.)
- **Data Dashboard:** Provides a structured summary for training analysis. (Relatively stable.)
- **Error Handling:** Ensures robustness when handling missing or invalid data. (Requires careful testing to cover all edge cases.)

- **Google Sheets Integration:** Reads data from `CompleteProgram` and `Maxes` tabs. (Rarely Updated)
- **Dynamic Weight Assignments:** Calculates weights using fitted multipliers. (Actively Maintained)
- **Confidence-Based Simulations:** Estimates Functional Max Strength conservatively. (Actively Maintained)
- **Functional Max Calculations:** Currently in development.
- **Data Dashboard:** Provides a structured summary for training analysis.
- **Error Handling:** Ensures robustness when handling missing or invalid data.
- **Google Sheets Integration:** Reads data from `CompleteProgram` and `Maxes` tabs. (Stable)
- **Dynamic Weight Assignments:** Calculates weights using fitted multipliers. (Stable)
- **Confidence-Based Simulations:** Estimates Functional Max Strength conservatively. (Under Review)
- **Data Dashboard:** Provides a structured summary for training analysis. (Stable)
- **Error Handling:** Ensures robustness when handling missing or invalid data. (In Progress)

- **Google Sheets Integration:** Reads data from `CompleteProgram` and `Maxes` tabs.
- **Dynamic Weight Assignments:** Calculates weights using fitted multipliers.
- **Confidence-Based Simulations:** Estimates Functional Max Strength conservatively.
- **Data Dashboard:** Provides a structured summary for training analysis.
- **Error Handling:** Ensures robustness when handling missing or invalid data.

## Installation

### Requirements
- **Python version:** 3.8 or newer is recommended.
- **Operating System:** Tested on Linux and macOS, compatibility with Windows not fully verified.
- **Additional tools:** Google Cloud SDK for generating service account credentials.

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. [Set up your credentials](#credentials-setup) for accessing Google Sheets.

## Credentials Setup

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

## Workflow

This workflow is typically initiated through manual runs, but future iterations may incorporate scheduled tasks and external triggers. For example, Google Sheet data input could automatically prompt data processing and simulation.

1. **Load Data:** Pulls lifting data from Google Sheets.
2. **Merge Data:** Aligns `CompleteProgram` and `Maxes` for calculations.
3. **Fit Multipliers:** Generates weight assignment formulas per exercise.
4. **Calculate Weights:** Computes and formats assigned weights.
5. **Simulate & Estimate Maxes:** Uses a conservative model for missing maxes.
6. **Generate Dashboard:** Summarizes key outputs for analysis.

## Project Structure

- **`execution/`**: Core scripts for processing lifting programs and handling data.
- **`data/`**: Local storage for temporary files, processed data, or caches.
- **`docs/`**: Documentation, including guides and references.
- **`tests/`**: Unit tests and other testing scripts.
- **`archive/`**: Deprecated or experimental files no longer in active use.
- **`scripts/`**: Standalone utility scripts (e.g., `fetch_github_project.py`).

## Key Project Inputs

**CompleteProgram:**
| Exercise          | Week # | Set # | Reps | Multiplier |
|-------------------|--------|-------|------|------------|
| Barbell Squat     | 1      | 1     | 5    | 0.8        |
| Bench Press       | 1      | 1     | 5    | 0.85       |

Expected ranges and values:
- **Reps:** Typically between 1 and 10.
- **Multiplier:** Typically between 0.1 and 1.
- **Week #:** Should be 1, 2, or 3.

**Maxes:**
| Player            | Core Lift      | Max |
|-------------------|----------------|-----|
| John Doe          | Barbell Squat  | 300 |
| Jane Doe          | Bench Press    | 200 |

Expected ranges and values:
- **Max:** Typical values range from 100 to 500 pounds.
- **Core Lift:** Common examples include "Barbell Squat," "Bench Press," and "Deadlift."
- **Player:** Any non-empty string representing a player's name.

**CompleteProgram:**

| Exercise      | Week # | Set # | Reps | Multiplier |
| ------------- | ------ | ----- | ---- | ---------- |
| Barbell Squat | 1      | 1     | 5    | 0.8        |
| Bench Press   | 1      | 1     | 5    | 0.85       |

**Maxes:**

| **Player** | **Core Lift** | **Max** |
| ---------- | ------------- | ------- |
| John Doe   | Barbell Squat | 300     |
| Jane Doe   | Bench Press   | 200     |

## Key Project Outputs

**AssignedWeights Validation:**
Each assigned weight is verified against the player’s tested max and a calculated multiplier. Invalid entries (e.g., missing maxes or unexpected column formats) are flagged and excluded from calculations. This process ensures that only valid weights are saved and written to the output tables.

**SimulatedData Validation:**
The simulation process uses statistical models to create realistic variations in reps and weights. If any calculated values fall outside predefined thresholds, they are adjusted or excluded, ensuring that the final simulated data is reliable and within expected ranges.

**AssignedWeights (Google Sheet):**

| **Player** | **Exercise**  | **Multiplier** | **Relevant Core** | **Assigned Weight** |
| ---------- | ------------- | -------------- | ----------------- | ------------------- |
| John Doe   | Barbell Squat | 0.8            | Barbell Squat     | 240                 |
| Jane Doe   | Bench Press   | 0.85           | Bench Press       | 170                 |

**SimulatedData (Google Sheet):**

| **Player** | **Exercise**  | **Assigned Reps** | **Assigned Weight** | **Simulated Reps** | **Simulated Weight** |
| ---------- | ------------- | ----------------- | ------------------- | ------------------ | -------------------- |
| John Doe   | Barbell Squat | 5                 | 240                 | 5                  | 245                  |
| Jane Doe   | Bench Press   | 5                 | 170                 | 5                  | 175                  |

## Usage

Run the main script:

```bash
python execution/main.py
```

## FAQ & Troubleshooting

### Common Questions
- **Q:** How do I handle authentication errors with Google Sheets?  
  **A:** Make sure the credentials file is in the correct directory and that the service account has access to the target spreadsheet.

- **Q:** What do I do if there are missing maxes?  
  **A:** The script assigns “NRM” (No Relevant Max) and calculates weights accordingly. Check your source data to ensure it’s complete.

- **Q:** Why is the script failing at the merge step?  
  **A:** Verify that the `CompleteProgram` and `Maxes` tabs have consistent column names and that the data types match.

### Additional Tips
- **Logging:** Check the log files in the `logs/` directory for detailed error messages and warnings.
- **Dependencies:** Ensure that all dependencies listed in `requirements.txt` are installed and up-to-date.
- **Environment Variables:** Confirm that all required environment variables are set correctly before running any scripts.

- **Q:** Where can I find the test results and how should I interpret them?  
  **A:** Test results are displayed in the GitHub Actions tab under the relevant workflow run. A green checkmark indicates that all tests passed, while a red X means at least one test failed. For more details, click on the workflow run to see the logs. The logs show which specific tests failed and provide the output messages to help diagnose the issue. Common indicators include assertion errors or missing data. This information can guide you on what to fix before submitting a pull request.

- **Q:** What are common scenarios that new contributors might encounter?  
  **A:**
  - **Data Schema Changes:** A new contributor might add columns to the `CompleteProgram` or `Maxes` sheets. Refer to the `data_merging.py` file to ensure the new columns are handled correctly.
  - **New Exercises:** Adding new exercises may require updates to the multiplier fitting logic. Check `multiplier_fitting.py` for how current exercises are mapped and ensure the new exercises have appropriate multipliers assigned.
  - **API Credential Updates:** If the service account keys expire or are rotated, a new contributor will need to replace the credentials and confirm that `config/client_secret.json` is updated and accessible.

- **Q:** Where should I start when addressing a new Issue?  
  **A:** Start by reviewing the Key Project Inputs section. Understanding the format and expected data ranges can help you pinpoint where things might be going wrong. If the issue is related to logic or calculations, check the relevant test files under `tests/`. For example, if the problem involves weight assignments, `test_weight_assignments.py` would be a good place to look. If the issue pertains to simulations, refer to `test_simulation.py`.

- **Q:** What error codes might I encounter and what do they mean?  
  **A:** Common error codes and messages include:
  - **403 Forbidden:** This often means that the service account credentials do not have access to the target Google Sheets file. Double-check the sharing settings in Google Sheets.
  - **404 Not Found:** This typically occurs if the specified sheet name does not exist. Make sure that the names of the tabs in the Google Sheets match exactly what the script expects.
  - **ValueError: Cannot merge DataFrames with overlapping column names:** This error suggests a column naming conflict. Check if columns in the CompleteProgram and Maxes sheets have identical names, and if so, rename them or adjust the merge logic.
  - **KeyError: 'Tested Max':** This indicates that the expected column is missing in the input data. Ensure that the Maxes tab includes a column for "Tested Max" with data for each relevant player.

These error codes and messages help identify the root cause and guide the resolution process.

- **Q:** How is data validated before calculations?  
  **A:** Data validation occurs primarily in the `data_processing.py` script. This includes:
  - **Flattening Core Maxes:** The core maxes are restructured into a normalized format, ensuring each player’s data aligns correctly.
  - **Generating a Repeated Program DataFrame:** The program information is expanded for all players, with checks to confirm consistent column structure and appropriate record counts.
  - **Logging and Verification:** The script logs any inconsistencies or unexpected data types, allowing errors to be identified early. Columns are validated to match the expected schema before merging data from different sources.

These preprocessing steps ensure that the input data is clean and structured properly before calculations or simulations begin.

- **Q:** What are some common edge cases and how are they handled?  
  **A:** Common edge cases include:
  - **Missing Player Maxes:** When a player lacks a recorded max for a key lift, the program assigns "NRM" and excludes them from simulations involving that lift.
  - **Duplicate Column Names:** If the `CompleteProgram` or `Maxes` data contains duplicate column headers, the merge logic automatically renames them to avoid overwriting data.
  - **Inconsistent Multipliers:** If an exercise multiplier is unexpectedly high or low, the code checks for input anomalies and flags the row for review before applying it to weight calculations.
  - **Invalid Data Types:** When the input data contains strings where numbers are expected (e.g., "N/A" in a numeric column), the program either converts them to default values or excludes them from calculations to ensure smooth processing.

- **Q:** What kind of issues arise most frequently?  
  **A:** Common issues include:
  - Ensuring data consistency when importing from Google Sheets.
  - Resolving incorrect or missing multiplier values for certain exercises.
  - Adjusting the simulation logic to more accurately reflect real-world performance.

- **Q:** What are some example fixes from past issues?  
  **A:** Past issues and resolutions include:
  - Adjusting the merge logic to handle duplicate column names and ensure proper alignment of program and max data.
  - Adding error handling to skip invalid entries while continuing to process valid data.
  - Implementing checks to confirm that all required API credentials are present before running the script.

- **Q:** How can I ensure that the simulated max values are reasonable?  
  **A:** If the simulated values seem off, verify that the underlying statistical distributions match the expected performance range. Also ensure that the input data is free from outliers or anomalies.

- **Q:** How do I handle authentication errors with Google Sheets?\
  **A:** Make sure the credentials file is in the correct directory and that the service account has access to the target spreadsheet.

- **Q:** What do I do if there are missing maxes?\
  **A:** The script assigns “NRM” (No Relevant Max) and calculates weights accordingly. Check your source data to ensure it’s complete.

- **Q:** Why is the script failing at the merge step?\
  **A:** Verify that the `CompleteProgram` and `Maxes` tabs have consistent column names and that the data types match.

## Pull Request Guidelines

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

### Required Tests Before Submitting a PR
Contributors must run the following tests to ensure consistency and functionality:
- **Unit Tests:**
  - `test_weight_assignments.py`: Validates that assigned weights are correct based on input maxes and multipliers.
  - `test_simulation.py`: Ensures that simulated reps and weights fall within expected ranges.
- **Integration Tests:**
  - `test_data_processing.py`: Confirms that input data is processed and structured correctly.
  - `test_data_merging.py`: Verifies that program data and maxes merge as intended without errors.
- **Validation Tests:**
  - `test_google_sheets_utils.py`: Checks that data is correctly read from and written to Google Sheets.

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

### How to Run Tests

To verify functionality, run the full test suite:
```bash
pytest tests/ --disable-warnings
```

If the issue affects weight assignments, prioritize:
- **test_weight_assignments.py:** Validates that assigned weights are correct based on input maxes and multipliers.

If the issue involves data simulation, start with:
- **test_simulation.py:** Ensures that simulated reps and weights fall within expected ranges.

For data merging or processing errors, focus on:
- **test_data_merging.py:** Verifies that program data and maxes merge as intended without errors.
- **test_data_processing.py:** Confirms that input data is processed and structured correctly.

These targeted tests can help you quickly isolate the problem before running the entire suite. Before Submitting a PR
Before submitting a pull request, ensure that all tests pass by running the test suite:
```bash
pytest tests/ --disable-warnings
```
Check the output for any failing tests and address those issues. This helps maintain a stable codebase and ensures that new changes do not introduce regressions.

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

### How to Link Issues
To link a pull request to an issue, include keywords like "Fixes #ISSUE_NUMBER" or "Closes #ISSUE_NUMBER" in the PR description. This will automatically close the issue when the PR is merged, ensuring that completed tasks are properly tracked and tied to their corresponding issues.

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

## Fixes & Updates

**Full Test Suite Passing (Latest Fixes):**

- \*\*Resolved \*\***`KeyError: 'Tested Max'`** in `test_assign_weights.py` by ensuring `"Tested Max"` is included in test data.
- **Fixed dtype compatibility issue** with `"NRM"` assignments in Pandas.
  - Explicitly converted `"Assigned Weight"` to `object` before assigning `"NRM"`.
- **Corrected ************`write_to_google_sheet()`************ behavior** to properly append rows even when creating a new worksheet.
- **Updated ************`test_google_sheets_utils.py`************ mocking** to properly simulate worksheet creation and prevent false failures.
- **Ensured ************`fit_multipliers()`************ accepts DataFrame input** instead of incorrectly receiving a dictionary.

## How to Run Tests

To verify functionality, run:

```bash
pytest tests/ --disable-warnings
```

## Maintainer

**Ricky St. Augustine**\
For questions, reach out via GitHub Issues.

