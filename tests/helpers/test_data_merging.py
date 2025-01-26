# tests/helpers/test_data_merging.py
import pytest
from execution.helpers.data_merging import merge_data

def test_merge_data():
    """Basic test for merge_data function."""
    result = merge_data()  # Call function
    assert isinstance(result, dict)  # Adjust based on actual output type

