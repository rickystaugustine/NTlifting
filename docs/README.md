# Lifting Program Automation

## 📌 Overview
This project automates the process of assigning lifting weights for a high school football strength program. It integrates data from Google Sheets, calculates assigned weights based on tested maxes and multipliers, and simulates missing data for a conservative Functional Max Strength estimate.

## 🚀 Features
- **Google Sheets Integration**: Reads data from `CompleteProgram` and `Maxes` tabs.
- **Dynamic Weight Assignments**: Calculates weights using fitted multipliers.
- **Confidence-Based Simulations**: Estimates Functional Max Strength conservatively.
- **Data Dashboard**: Provides a structured summary for training analysis.
- **Error Handling**: Ensures robustness when handling missing or invalid data.

## 🔧 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Google Sheets API authentication:
   - Add your credentials JSON file to the project directory.
   - Ensure access to the relevant Google Sheets.

## 📊 Workflow
1. **Load Data** – Pulls lifting data from Google Sheets.
2. **Merge Data** – Aligns `CompleteProgram` and `Maxes` for calculations.
3. **Fit Multipliers** – Generates weight assignment formulas per exercise.
4. **Calculate Weights** – Computes and formats assigned weights.
5. **Simulate & Estimate Maxes** – Uses a conservative model for missing maxes.
6. **Generate Dashboard** – Summarizes key outputs for analysis.

## 🏗 Project Structure
```
/lifting-program-automation
│-- /src                 # Core scripts
│-- /data                # Local storage for processing (if needed)
│-- /docs                # Documentation
│-- /notebooks           # Jupyter notebooks (for analysis/testing)
│-- /tests               # Unit tests
│-- requirements.txt     # Python dependencies
│-- README.md            # Project overview
```

## 🔄 Usage
Run the main script:
```bash
python src/main.py
```

## 📌 Pull Request Naming Convention
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

## 📜 License
This project is licensed under the MIT License.

## 🤝 Contributing
1. Fork the repo.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add new feature"`).
4. Push to your branch (`git push origin feature-name`).
5. Open a Pull Request.

## 📝 Pull Request Guidelines
To maintain consistency and readability, all pull requests must follow a structured format.

📌 **Follow the PR Template:**  
- The PR template is located at [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md).
- Use the correct naming convention:   [TYPE] #ISSUE_NUMBER - Short Description
- **Examples:**
- [BUGFIX] #7 - Fix Google Sheets Data Loading
- [FEATURE] #15 - Implement Weight Progression Model
- [DOCS] #22 - Update README Formatting
- - Before submitting, ensure:
- [ ] The PR is linked to an issue.
- [ ] Code changes are tested.
- [ ] No breaking changes are introduced.

## 🏆 Maintainer
**Ricky St. Augustine**  
For questions, reach out via GitHub Issues.
