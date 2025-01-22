import os
import shutil
import re
from datetime import datetime
import sys
import os

# Ensure Python can find core_scripts and helpers
sys.path.append("/Users/ricky.staugustine/Documents/Programming/GitHub/NTlifting/core_scripts")
sys.path.append("/Users/ricky.staugustine/Documents/Programming/GitHub/NTlifting/core_scripts/helpers")


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def move_files_by_category(base_dir):
    categories = {
        "core": ["FunctionalMaxCalculations", "MainModel", "LiftingCardCalculations", "FixWeightDiffs", "GoogleLiftingCardSheet"],
        "helpers": ["helpers", "Modularization", "HighHangPullFix", "HexBarDLfix"],
        "analysis": ["AnalyzeResidualPatterns", "SimulatedDataAnalysis", "CleanTestModel", "ContextualizedCurveFitting", "DynamicHybridModel", "FunctionalMaxHybridModel", "FunctionalMaxReverseEngineering", "FunctionalMaxCalculationTesting", "FunctionalMaxWithHHPfix"],
        "archive": ["populate_card"]
    }
    
    for category, keywords in categories.items():
        category_path = os.path.join(base_dir, category)
        ensure_directory_exists(category_path)
        
        for file in os.listdir(base_dir):
            if file.endswith(".py"):
                for keyword in keywords:
                    if re.search(keyword, file, re.IGNORECASE):
                        shutil.move(os.path.join(base_dir, file), os.path.join(category_path, file))
                        print(f"Moved {file} to {category_path}")
                        break

def find_unused_scripts(base_dir, days_threshold=180):
    archive_path = os.path.join(base_dir, "archive")
    ensure_directory_exists(archive_path)
    
    now = datetime.now().timestamp()
    for file in os.listdir(base_dir):
        if file.endswith(".py"):
            file_path = os.path.join(base_dir, file)
            last_modified = os.path.getmtime(file_path)
            age_days = (now - last_modified) / (60 * 60 * 24)
            if age_days > days_threshold:
                shutil.move(file_path, os.path.join(archive_path, file))
                print(f"Archived {file} (Last modified: {age_days:.0f} days ago)")

def check_missing_dependencies(base_dir):
    dependencies = set()
    scripts = [f for f in os.listdir(base_dir) if f.endswith(".py")]
    
    for script in scripts:
        with open(os.path.join(base_dir, script), "r") as f:
            for line in f:
                match = re.match(r"^(?:import|from) ([a-zA-Z0-9_]+)", line)
                if match:
                    dependencies.add(match.group(1))
    
    print("\nChecking for missing dependencies:")
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            print(f"⚠️ Missing dependency: {dep}")

def main():
    base_dir = "core_scripts"  # Adjust if needed
    move_files_by_category(base_dir)
    find_unused_scripts(base_dir)
    check_missing_dependencies(base_dir)
    print("\n✅ Organization complete!")

if __name__ == "__main__":
    main()

