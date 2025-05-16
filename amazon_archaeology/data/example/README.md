# Example Dataset

This directory contains sample data for testing the Amazon Archaeological Discovery Tool without needing to download large datasets or set up API keys.

## Contents

- `lidar_sample.tif`: Sample LiDAR data for a small region of the Amazon
- `satellite_sample.tif`: Sample Sentinel-2 imagery for the same region
- `historical_text.txt`: Example historical text with potential archaeological site references
- `known_sites.geojson`: GeoJSON file with known archaeological sites in the region

## Using Example Data

To run the tool with sample data:

```bash
python run.py --use-example-data
```

Or access through the API:

```bash
python run.py --api-mode
```

Then in your web browser:
```
http://localhost:8000/example
```

## Data Sources

This example data is derived from:
- LiDAR: Simplified from OpenTopography SRTM data
- Satellite: Simplified from Sentinel-2 open data
- Historical: Based on public domain expedition accounts
- Known sites: Based on published archaeological surveys 