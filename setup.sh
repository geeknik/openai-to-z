#!/bin/bash
# Setup script for the Amazon Archaeological Discovery Tool

# Print colored text
print_green() {
    echo -e "\033[0;32m$1\033[0m"
}

print_yellow() {
    echo -e "\033[0;33m$1\033[0m"
}

print_red() {
    echo -e "\033[0;31m$1\033[0m"
}

print_blue() {
    echo -e "\033[0;34m$1\033[0m"
}

# Display header
print_blue "================================================="
print_blue "    Amazon Archaeological Discovery Tool Setup    "
print_blue "================================================="
echo ""

# Create necessary directories
print_green "Creating required directories..."
mkdir -p amazon_archaeology/data/lidar
mkdir -p amazon_archaeology/data/satellite
mkdir -p amazon_archaeology/data/visualizations
mkdir -p amazon_archaeology/data/cache
mkdir -p amazon_archaeology/data/example

# Create Python virtual environment
if [ ! -d "venv" ]; then
    print_green "Setting up Python virtual environment..."
    python3 -m venv venv
    
    if [ $? -ne 0 ]; then
        print_red "Error: Failed to create virtual environment."
        print_yellow "Please ensure you have Python 3.8+ installed with venv support."
        exit 1
    fi
else
    print_yellow "Using existing virtual environment."
fi

# Activate virtual environment
print_green "Activating virtual environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    print_red "Error: Could not find activation script for virtual environment."
    exit 1
fi

# Install dependencies
print_green "Installing required Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if lxml is installed (required for KML conversion)
print_green "Installing optional dependencies for KML conversion..."
pip install lxml

# Check for geospatial dependencies
print_green "Installing geospatial dependencies..."
print_yellow "Note: Some macOS users may experience issues with Fiona's 'path' attribute."
print_yellow "The system includes a fallback mechanism if these libraries aren't available."

# On macOS, use brew to install GDAL first if it's available
if [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v brew >/dev/null 2>&1; then
        print_yellow "macOS detected. Installing GDAL via Homebrew for better compatibility..."
        brew install gdal
    else
        print_yellow "macOS detected but Homebrew not found. Consider installing GDAL manually for better compatibility."
    fi
fi

# Now install Python packages
if pip install -r amazon_archaeology/requirements-geo.txt; then
    print_green "Successfully installed geospatial dependencies."
else
    print_yellow "Warning: Could not install some geospatial dependencies."
    print_yellow "KML conversion will use a simplified method."
    print_yellow "See README for more information on optional dependencies."
fi

# Set up environment file if not exists
if [ ! -f "amazon_archaeology/.env" ]; then
    print_green "Creating environment file..."
    cp env.example amazon_archaeology/.env
    print_yellow "Please edit amazon_archaeology/.env with your API keys."
fi

# Final instructions
echo ""
print_green "Setup complete!"
print_blue "================================================="
print_blue "                  Next Steps                     "
print_blue "================================================="
print_yellow "1. Activate the virtual environment if not using this shell:"
echo "   source venv/bin/activate  # Linux/Mac"
echo "   .\\venv\\Scripts\\activate.bat  # Windows"
echo ""

print_yellow "2. Edit your API keys (if needed):"
echo "   nano amazon_archaeology/.env"
echo ""

print_yellow "3. Run with example data to test:"
echo "   python amazon_archaeology/run.py --use-example-data"
echo ""

print_yellow "4. Explore Amazon LiDAR datasets:"
echo "   python amazon_archaeology/run.py list-amazon-datasets"
echo ""

print_yellow "5. View documentation:"
echo "   less amazon_archaeology/docs/AMAZON_LIDAR.md"
echo ""

print_green "Happy archaeological discovery!"
echo "" 