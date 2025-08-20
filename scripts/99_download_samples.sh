#!/bin/bash
# Setup script for AI Code Reviewer - using existing data

set -e

echo "Setting up AI Code Reviewer with existing data..."

# Check if we're in the right directory structure
if [ ! -d "data/raw/bugsinpy/projects" ] || [ ! -d "data/raw/quixbugs/python_programs" ] || [ ! -d "data/raw/quixbugs/correct_python_programs" ]; then
    echo "Error: Required data directories not found"
    echo "Please ensure you're running this from the project root directory"
    echo "and that the data has been properly organized in data/raw/"
    exit 1
fi

# Create data directories
mkdir -p data/interim
mkdir -p data/processed

echo "Data setup completed!"
echo ""
echo "Data locations:"
echo "  BugsInPy: data/raw/bugsinpy"
echo "  QuixBugs: data/raw/quixbugs"
echo ""
echo "Next steps:"
echo "  1. Run: make bugsinpy"
echo "  2. Run: make quixbugs"
echo "  3. Run: make split"
echo "  4. Run: make stats"
