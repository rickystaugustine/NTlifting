import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, ".."))

print(f"✅ Updated sys.path: {sys.path}")
