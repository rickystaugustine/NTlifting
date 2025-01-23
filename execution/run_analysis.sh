#!/bin/bash

export PYTHONPATH=$(pwd)/execution

echo "🔹 Running Assigned Weights Review..."
python execution/analysis/assigned_weights_review.py | tee execution/logs/assigned_weights_review.log || echo "⚠️ Assigned Weights Review Failed"

echo "🔹 Running Fitting Review..."
python execution/analysis/fitting_review.py | tee execution/logs/fitting_review.log || echo "⚠️ Fitting Review Failed"

echo "🔹 Running Simulated Data Review..."
python execution/analysis/simulated_data_review.py | tee execution/logs/simulated_data_review.log || echo "⚠️ Simulated Data Review Failed"

echo "🔹 Running Functional Max Review..."
python execution/analysis/functional_max_review.py | tee execution/logs/functional_max_review.log || echo "⚠️ Functional Max Review Failed"

echo "🔹 Running Functional Max Validation..."
python execution/analysis/functional_max_validation.py | tee execution/logs/functional_max_validation.log || echo "⚠️ Functional Max Validation Failed"

echo "✅ All analyses completed!"
