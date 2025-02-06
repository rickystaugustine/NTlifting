# Contributing Guidelines

Thank you for considering contributing to this project! This guide will help you get started and understand the process of submitting changes.

## 🛠️ Setting Up Your Development Environment
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

## 📦 Installing Dependencies
- **Core dependencies:** Located in `requirements.txt`. These are required to run the main application.
- **Development dependencies:** Located in `requirements-dev.txt`. These are needed for testing, code quality checks, and documentation.

Make sure both files are present in the project’s root directory.

## 🔄 Workflow
1. **Create a new branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes:**
   - Add features, fix bugs, or improve documentation.
3. **Run tests:**
   ```bash
   pytest tests/ --disable-warnings
   ```
4. **Check code quality:**
   - Consider running linters or formatters if used.
   
## 📋 Pull Request Process
1. **Follow naming conventions:**
   ```
   [TYPE] #ISSUE_NUMBER - Short Description
   ```
   **Example:**
   ```
   [FEATURE] #42 - Add weight progression model
   ```
   **PR Types:**
   - `[FEATURE]` → New features
   - `[BUGFIX]` → Bug fixes
   - `[REFACTOR]` → Code cleanups
   - `[DOCS]` → Documentation updates

2. **Link to an issue:**
   - Always link the pull request to a GitHub issue if applicable. Use the syntax:
     ```
     Closes #ISSUE_NUMBER
     ```

3. **Provide a summary:**
   - Describe what the pull request does.
   - List any changes and testing steps.

4. **Submit for review:**
   - Push your branch:
     ```bash
     git push origin feature/your-feature-name
     ```
   - Open a pull request on GitHub.

## ✅ Testing and Review
- Ensure all tests pass before submitting your PR.
- Confirm that no breaking changes are introduced.
- Reviewers will verify the changes meet the project’s standards and follow best practices.

## 📚 Additional Notes
- **Continuous Integration:** The project uses GitHub Actions workflows to run tests on each pull request.
- **Documentation:** Update `README.md` and any other relevant documentation if your changes alter usage or functionality.
- **Code Coverage:** If possible, ensure new code is covered by tests.

---

Following these guidelines will help maintain a consistent and efficient workflow for all contributors, ensuring the project’s ongoing quality and stability.


