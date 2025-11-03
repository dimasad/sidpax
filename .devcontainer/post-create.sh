#!/bin/bash
set -e  # Exit if any command fails

echo "Creating virtual environment..."
python -m venv .venv

echo "Activating the virtual environment..."
source .venv/bin/activate

echo "Installing the package with dependencies..."
pip install -e ".[devextra,test]"
