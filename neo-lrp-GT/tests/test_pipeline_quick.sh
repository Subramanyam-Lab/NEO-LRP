#!/bin/bash
# Quick Pipeline Test for NEO-LRP-GT
# ==================================
#
# This script tests the complete pipeline quickly:
# 1. Checks dependencies
# 2. Tests data loading from test_data.h5
# 3. Runs training for 3 epochs
#
# Usage: ./test_pipeline_quick.sh

set -e  # Exit on any error

echo "🚀 NEO-LRP-GT Quick Pipeline Test"
echo "================================="
echo ""

# Check if we're in the right directory
if [[ ! -f "../train.py" ]] || [[ ! -f "../test_data.h5" ]]; then
    echo "❌ Error: Please run this script from the neo-lrp-GT/tests directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected files: ../train.py, ../test_data.h5"
    exit 1
fi

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not detected. Attempting to activate..."
    if [[ -f "../../venv/bin/activate" ]]; then
        source ../../venv/bin/activate
        echo "✅ Activated virtual environment"
    else
        echo "❌ Virtual environment not found at ../../venv/"
        echo "   Please run: python -m venv ../../venv && source ../../venv/bin/activate"
        exit 1
    fi
fi

echo "🧪 Running simple training test..."
echo ""

python simple_test.py

echo ""
echo "🎉 Pipeline test completed!"
echo ""
echo "📋 What was tested:"
echo "   ✅ Dependencies (torch, torch-geometric, h5py, etc.)"
echo "   ✅ Data loading from test_data.h5"
echo "   ✅ Neural network training (3 epochs)"
echo "   ✅ Model saving to model_state/"
echo ""
echo "📋 To run full tests:"
echo "   • cd .. && python train.py --data-file test_data.h5    # Full training"
echo "   • cd .. && python train.py --data-file your_data.h5    # With your data"