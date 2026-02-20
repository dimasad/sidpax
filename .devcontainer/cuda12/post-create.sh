#!/bin/bash
set -e  # Exit if any command fails

echo "🛠️ Installing cyipopt build dependencies..."
sudo apt update
sudo apt install coinor-libipopt-dev -y

# Check for existing virtual environment
if [ -d ".venv" ]; then
    n=1
    # Find the next available .venv_old_N directory
    while [ -d ".venv_old_$n" ]; do
        n=$((n + 1))
    done

    new_name=".venv_old_$n"
    echo "⚠️  Found existing .venv directory."
    echo "   Moving it to $new_name ..."
    mv .venv "$new_name"
    echo "   (You can delete $new_name later if you don't need it.)"
fi

echo "🐍 Creating new virtual environment..."
python -m venv .venv

echo "💡 Activating the virtual environment..."
source .venv/bin/activate

echo "⬆️  Upgrading pip..."
python -m pip install --upgrade pip

echo "📦 Installing jax with GPU support..."
pip install -U "jax[cuda12]"

echo "📦 Installing the package with dependencies..."
pip install -e ".[devextra,test]"

echo "✅ Dev Container setup complete!"
