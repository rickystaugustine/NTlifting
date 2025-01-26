import pandas as pd
import logging
import sys
import os
# Explicitly add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("✅ Root directory added to sys.path:", project_root)

from execution.helpers.google_sheets_utils import write_to_google_sheet, read_google_sheets

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Define the test Google Sheet details
TEST_SHEET_NAME = "After-School Lifting"
TEST_WORKSHEET_NAME = "Test"

# ✅ Step 1: Read from the test sheet
df = read_google_sheets(TEST_SHEET_NAME, TEST_WORKSHEET_NAME)
print("📥 Read DataFrame from Google Sheets:")
print(df)

# ✅ Step 2: Modify the DataFrame for testing
test_data = pd.DataFrame({
    "Player": ["Wolfe", "Gebhart", "Davis"],
    "Tested Max": ["NRM", 75, 115],
    "Assigned Weight": [0, 50, 90]
})

# ✅ Step 3: Write to the test sheet
write_to_google_sheet(TEST_SHEET_NAME, TEST_WORKSHEET_NAME, test_data)

# ✅ Step 4: Read back the written data to verify
df_after = read_google_sheets(TEST_SHEET_NAME, TEST_WORKSHEET_NAME)
print("📤 DataFrame after Writing:")
print(df_after)

