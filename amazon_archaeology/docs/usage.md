# Amazon Archaeological Discovery Tool - Usage Guide

This document provides detailed instructions for using the Amazon Archaeological Discovery Tool to detect potential archaeological sites in LiDAR and satellite imagery.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Basic Usage](#basic-usage)
4. [Command-Line Options](#command-line-options)
5. [Working with Example Data](#working-with-example-data)
6. [Analyzing Custom Regions](#analyzing-custom-regions)
7. [Visualization](#visualization)
8. [API Mode](#api-mode)
9. [Cost Optimization Strategies](#cost-optimization-strategies)
10. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8+
- pip package manager
- [Optional] GDAL libraries for advanced geospatial processing

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/amazon-archaeology.git
   cd amazon-archaeology
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   ```bash
   cp env.example .env
   ```
   Then edit the `.env` file with your API keys and configuration settings.

## Configuration

The tool is configured through a combination of environment variables and command-line arguments. Key configuration parameters include:

- **OPENAI_API_KEY**: Your OpenAI API key for AI-assisted feature validation
- **DATA_DIR**: Base directory for data storage (defaults to amazon_archaeology/data/)
- **INITIAL_REGION_BOUNDS**: Default region to analyze if none specified
- **SAMPLE_RESOLUTION**: Resolution to sample data (in meters)
- **ENABLE_CACHE**: Whether to enable caching (set to "true" or "false")

These settings can be specified in your `.env` file. For more detailed settings, see [config.py](../src/config.py).

## Basic Usage

The simplest way to run the tool is:

```bash
python amazon_archaeology/run.py --use-example-data
```

This will:
1. Load example data (or generate it if none exists)
2. Run the feature detection pipeline
3. Merge nearby detections
4. Save results to a GeoJSON file
5. Print a summary of findings

## Command-Line Options

The tool provides several command-line options:

| Option | Default | Description |
|--------|---------|-------------|
| `--bounds` | From config | Region bounds in format 'west,south,east,north' |
| `--min-size` | 80 | Minimum feature size in meters |
| `--max-size` | 500 | Maximum feature size in meters |
| `--merge-distance` | 100.0 | Distance in meters to merge nearby features |
| `--sensitivity` | 0.5 | Detection sensitivity (0-1) |
| `--output` | Auto-generated | Output path for detected features (GeoJSON) |
| `--clear-cache` | False | Clear cache before running |
| `--api-mode` | False | Run in API server mode |
| `--use-example-data` | False | Use example data instead of fetching new data |
| `--visualize` | False | Create visualization of results |

## Working with Example Data

The tool includes example data for testing:

```bash
python amazon_archaeology/run.py --use-example-data --visualize
```

This will:
1. Load or generate synthetic test data
2. Run the detection pipeline
3. Create interactive visualizations
4. Save results to the data directory

The example data includes:
- Synthetic LiDAR data with geometric features
- Synthetic satellite imagery with vegetation anomalies
- Historical text records mentioning sites
- GeoJSON of known archaeological sites

## Analyzing Custom Regions

To analyze a specific region, you can specify bounds:

```bash
python amazon_archaeology/run.py --bounds=-64.1,-3.1,-63.9,-2.9
```

The tool will:
1. Check for available data sources for the region
2. Download LiDAR and satellite data (if available)
3. Run the detection pipeline
4. Save and display results

For optimal results, specify a region no larger than 0.2° × 0.2° (approximately 20km × 20km at the equator).

## Visualization

The tool can generate interactive visualizations with the `--visualize` flag:

```bash
python amazon_archaeology/run.py --use-example-data --visualize
```

This creates:
1. An interactive HTML map with feature locations and metadata
2. Statistical plots showing feature distributions
3. Output links to access the visualizations

Visualizations are saved to the `data/visualizations/` directory with timestamped filenames.

### Interactive Map Features

The interactive map includes:
- Markers for each detected feature
- Color-coding based on confidence levels
- Popups with detailed feature information
- Optional heatmap of detection confidence
- Layer controls for different feature types
- Measurement tools for size estimation

### Statistical Plots

The tool generates several plots:
- Feature type distribution
- Confidence score distribution
- Size distribution of detected features
- Scatter plot of confidence vs. size

## API Mode

The tool can run as an API server for integration with other applications:

```bash
python amazon_archaeology/run.py --api-mode
```

Once running, access the API documentation at: http://localhost:8000/docs

The API provides endpoints for:
- `/detect`: Run detection on a specified region
- `/features`: Get previously detected features
- `/visualize`: Generate visualizations for features
- `/sources`: Get available data sources for a region

## Cost Optimization Strategies

The tool implements several cost-optimization strategies:

### Data Management

- **Caching**: Results are cached to prevent redundant processing. Clear the cache with `--clear-cache`.
- **Sampling**: Data is sampled at configurable resolutions to reduce processing requirements.
- **Incremental Processing**: Only the necessary data for a region is downloaded and processed.

### API Usage

- **Two-Stage Detection**: Uses inexpensive local algorithms for initial detection, with AI verification only for ambiguous cases.
- **Response Caching**: AI API responses are cached to prevent duplicate calls.
- **Parameter Tuning**: Adjust `--sensitivity` to control the detection threshold.

### Computational Resources

- **Local Processing**: Core processing runs locally to avoid cloud computing costs.
- **Selective Visualization**: Generate visualizations only when needed with the `--visualize` flag.

## Troubleshooting

### Common Issues

#### No Data Available

If you see "No LiDAR or SRTM data available for this region":
- Try a different region with better data coverage
- Use example data for testing: `--use-example-data`
- Check data source connections in your `.env` file

#### Memory Issues

If you encounter memory problems with large regions:
- Reduce the region size in `--bounds`
- Increase `SAMPLE_RESOLUTION` in your `.env` file
- Process in smaller tiles by running multiple commands for adjacent areas

#### API Connection Errors

If API connections fail:
- Verify your API keys in the `.env` file
- Check your internet connection
- Ensure any required services are accessible

### Logging

The tool creates detailed logs in `data/amazon_archaeology.log`. Check this file for diagnostic information if you encounter issues.

## Advanced Usage

### Creating Custom Test Data

You can create custom test data by modifying the example data generator in `run.py` or by placing your own GeoTIFF files in the appropriate directories:

- LiDAR data: `data/example/lidar_sample.tif`
- Satellite data: `data/example/satellite_sample.tif`

### Training Custom Models

The tool supports plugging in custom-trained models for feature detection. To implement:

1. Place model files in the `models/` directory
2. Update configuration to point to your models
3. Modify the detection functions to use your custom models

For more details, see the developer documentation 