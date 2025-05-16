#!/bin/bash
# Setup script for the Amazon Archaeological Discovery Tool

set -e  # Exit on any error

# Print header
echo "===================================================="
echo "Amazon Archaeological Discovery Tool - Setup Script"
echo "===================================================="
echo

# Create required directories
echo "Creating required directories..."
mkdir -p amazon_archaeology/data/lidar
mkdir -p amazon_archaeology/data/satellite
mkdir -p amazon_archaeology/data/cache
mkdir -p amazon_archaeology/data/visualizations
echo "✅ Directories created successfully"
echo

# Setup Python environment
echo "Setting up Python environment..."
if [ -x "$(command -v python3)" ]; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Check Python version
PY_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Detected Python version: $PY_VERSION"

# Create virtual environment
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv
echo "✅ Virtual environment created"

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi
echo "✅ Virtual environment activated"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Core dependencies installed"

# Ask about installing geo-specific dependencies
echo
echo "Do you want to install additional geo-specific dependencies for KML conversion?"
echo "These are required for Google Earth integration."
read -p "Install geo dependencies? (y/n): " INSTALL_GEO

if [[ "$INSTALL_GEO" == "y" || "$INSTALL_GEO" == "Y" ]]; then
    echo "Installing geo-specific dependencies..."
    pip install -r amazon_archaeology/requirements-geo.txt
    echo "✅ Geo dependencies installed"
else
    echo "Skipping geo dependencies. You can install them later with:"
    echo "pip install -r amazon_archaeology/requirements-geo.txt"
fi

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo
    echo "Setting up environment configuration..."
    cp env.example .env
    echo "✅ Environment configuration file created (.env)"
    echo "   Edit this file to add your API keys if needed"
fi

# Run a simple test to verify installation
echo
echo "Testing the installation..."
$PYTHON_CMD -c "
import sys
sys.path.insert(0, 'amazon_archaeology')
from src.config import DATA_DIR
from src.utils.cache import clear_cache
print('✅ Core modules loaded successfully')
print(f'✅ Data directory is set to: {DATA_DIR}')
"

# Final instructions
echo
echo "===================================================="
echo "Setup completed successfully! 🚀"
echo "===================================================="
echo
echo "To get started, follow these steps:"
echo
echo "1. Activate the virtual environment (if not already activated):"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "   source venv/Scripts/activate"
else
    echo "   source venv/bin/activate"
fi
echo
echo "2. Run the tool with example data:"
echo "   python amazon_archaeology/run.py analyze --use-example-data --visualize"
echo
echo "3. See the README.md for more usage examples"
echo
echo "Happy archaeological discovery!"
echo 