# tests/conftest.py
import sys
import os

# Add the project root directory to sys.path to enable absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

print(f"✅ Root directory added to sys.path: {sys.path[0]}")
