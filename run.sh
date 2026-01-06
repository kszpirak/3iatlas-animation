#!/bin/bash
# run.sh - Launch 3I/ATLAS animation
# Usage: ./run.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found."
    echo "Please run ./install.sh first."
    exit 1
fi

# Activate and run
source .venv/bin/activate
python main.py "$@"
