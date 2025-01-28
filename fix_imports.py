import os
import re

# Define the directory containing the Python files
HELPERS_DIR = "execution/helpers"

# Define the incorrect and correct import patterns
INCORRECT_IMPORT_PATTERN = re.compile(r"from helpers\.(\w+) import")
CORRECT_IMPORT_TEMPLATE = r"from execution.helpers.\1 import"

def fix_imports_in_file(file_path):
    """Replace incorrect imports in a given Python file."""
    with open(file_path, "r") as f:
        content = f.read()

    # Replace incorrect imports with the correct ones
    new_content = INCORRECT_IMPORT_PATTERN.sub(CORRECT_IMPORT_TEMPLATE, content)

    # Only write if changes were made
    if new_content != content:
        with open(file_path, "w") as f:
            f.write(new_content)
        print(f"✅ Fixed imports in: {file_path}")

def process_directory(directory):
    """Process all .py files in the specified directory."""
    for file_name in os.listdir(directory):
        if file_name.endswith(".py"):
            fix_imports_in_file(os.path.join(directory, file_name))

if __name__ == "__main__":
    process_directory(HELPERS_DIR)
    print("🎯 Import path fixes completed!")

