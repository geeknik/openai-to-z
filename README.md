# Amazon Archaeological Discovery Tool

A cost-efficient, AI-powered tool for discovering archaeological sites in the Amazon basin using satellite imagery, LiDAR data, and historical records. Built for the OpenAI to Z Challenge.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Using the Tool](#using-the-tool)
  - [Analyzing Regions](#analyzing-regions)
  - [Visualizing Results](#visualizing-results)
  - [Google Earth Integration](#google-earth-integration)
  - [API Mode](#api-mode)
- [Cost Optimization Strategies](#cost-optimization-strategies)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [License](#license)

## Overview

The Amazon Archaeological Discovery Tool uses a combination of digital remote sensing, machine learning, and historical data analysis to identify potential archaeological sites in the Amazon basin. It focuses on detecting geometric shapes, vegetation anomalies, and other features that may indicate ancient human settlements.

The tool is designed to be cost-effective, utilizing free and open-source data where possible and implementing intelligent caching and processing strategies to minimize API and computational costs.

## Key Features

- **Multi-source Detection**: Analyzes LiDAR, satellite imagery, and historical text for a comprehensive approach
- **Cost-efficient Implementation**: Uses tiered model approach and local processing to minimize costs
- **Interactive Visualization**: Creates maps and statistical plots of detected features
- **API Support**: Offers a RESTful API for integration with other applications
- **Reproducible Results**: Provides structured output and consistent methodologies
- **Customizable Parameters**: Allows tuning of detection sensitivity and feature sizes

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/amazon-archaeology.git
   cd amazon-archaeology
   ```

2. Run the setup script (recommended):
   ```bash
   # Unix/Linux/Mac
   ./setup.sh
   
   # Windows (in Git Bash or similar)
   bash setup.sh
   ```
   
   This script will:
   - Create required directories
   - Set up a Python virtual environment
   - Install dependencies
   - Create an environment file
   
   Or manually install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up environment variables:
   ```bash
   cp env.example .env
   ```
   Then edit the `.env` file with your API keys and configuration settings.

## Quick Start

Run the tool with example data to quickly test its functionality:

```bash
python amazon_archaeology/run.py --use-example-data --visualize
```

This will:
1. Load (or generate) example data
2. Detect potential archaeological features
3. Create visualizations
4. Display a summary of results

## Using the Tool

### Analyzing Regions

To analyze a specific region in the Amazon:

```bash
python amazon_archaeology/run.py --bounds=-64.1,-3.1,-63.9,-2.9
```

Command-line options include:

| Option | Purpose |
|--------|---------|
| `--bounds` | Region to analyze (west,south,east,north) |
| `--min-size` | Minimum feature size in meters |
| `--max-size` | Maximum feature size in meters |
| `--sensitivity` | Detection sensitivity (0-1) |
| `--merge-distance` | Distance to merge nearby features |
| `--visualize` | Create interactive visualizations |
| `--create-kml` | Generate KML files for Google Earth |
| `--output` | Custom output path for results |
| `--clear-cache` | Clear cached data before running |

### Visualizing Results

The `--visualize` flag generates:

1. An interactive HTML map with all detected features
2. Statistical plots showing feature distributions
3. Links to access these visualizations

Example visualization:
```bash
python amazon_archaeology/run.py --use-example-data --visualize --sensitivity 0.7
```

### Google Earth Integration

Convert detection results to KML format for visualization in Google Earth:

```bash
# Generate KML during analysis
python amazon_archaeology/run.py --bounds=-67.5,-10.1,-67.4,-10.0 --create-kml

# Convert existing GeoJSON results to KML
python amazon_archaeology/geojson_to_kml.py file path/to/features.geojson
```

The KML files can be opened directly in Google Earth Pro, allowing for:
- 3D visualization of archaeological sites in their geographical context
- Color-coded markers based on confidence levels
- Detailed information about each site when clicked
- Sharing findings with collaborators

See [KML Conversion Documentation](amazon_archaeology/docs/KML_CONVERSION.md) for detailed instructions.

### API Mode

Run as an API server for integration with other applications:

```bash
python amazon_archaeology/run.py --api-mode
```

Access the API documentation at: http://localhost:8000/docs

## Cost Optimization Strategies

This tool implements several strategies to minimize costs while maintaining performance:

### Data Management
- **Free Data Sources**: Exclusively uses open-source satellite imagery (Sentinel), LiDAR (OpenTopography), and elevation data (SRTM)
- **Caching System**: Prevents redundant downloads and processing
- **Resolution Control**: Configurable sampling resolution to balance detail with resource usage

### API Usage
- **Tiered Model Approach**:
  - Uses cheaper models (e.g., `o3-mini`) for initial screening
  - Only uses expensive models (e.g., `gpt-4`) for ambiguous cases
  - Response caching prevents duplicate API calls

### Computational Resources
- **Local Processing Pipeline**: Core image processing runs locally to avoid cloud compute costs
- **Lightweight Visualization**: Uses Folium (wrapper for Leaflet.js) instead of paid solutions
- **Incremental Region Analysis**: Analyzes targeted regions before scaling

## Project Structure

```
amazon_archaeology/
├── data/                  # Data storage
│   ├── example/           # Example data for testing
│   └── visualizations/    # Generated visualizations
├── docs/                  # Documentation
├── src/                   # Source code
│   ├── analysis/          # Feature detection algorithms
│   ├── api/               # API implementation
│   ├── preprocessing/     # Data fetching and preparation
│   ├── utils/             # Utility functions
│   └── visualization/     # Visualization tools
├── tests/                 # Automated tests
├── config.py              # Configuration settings
└── run.py                 # Main entry point
```

## Documentation

- [Usage Guide](amazon_archaeology/docs/usage.md): Detailed usage instructions
- [API Documentation](http://localhost:8000/docs): Interactive API documentation (when running in API mode)
- [Design Document](DESIGN.md): Original design specification
- [Implementation Details](IMPLEMENTATION.md): Implementation strategies and cost optimization
- [KML Conversion](amazon_archaeology/docs/KML_CONVERSION.md): Guide for exporting to Google Earth

## License

[MIT License](LICENSE) 