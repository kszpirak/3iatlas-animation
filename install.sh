#!/bin/bash
# install.sh - Set up Python environment for 3I/ATLAS animation
# Usage: ./install.sh

set -e  # Exit on error

echo "=== 3I/ATLAS Animation - Installation ==="
echo

# Check for Python 3
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python 3 not found. Please install Python 3.10+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PYTHON_VERSION"

# Create virtual environment
echo
echo "Creating virtual environment..."
$PYTHON_CMD -m venv .venv

# Activate venv
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo
echo "Installing dependencies..."
pip install -r requirements.txt

echo
echo "=== Installation complete! ==="
echo
echo "To run the simulation:"
echo "  ./run.sh"
echo
echo "Or manually:"
echo "  source .venv/bin/activate"
echo "  python main.py"
