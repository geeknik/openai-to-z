# GeoJSON to KML Conversion for Google Earth

The Amazon Archaeological Discovery Tool provides functionality to convert detection results (GeoJSON format) to KML files that can be visualized in Google Earth. This document explains how to use this feature.

## Quick Usage

### 1. Automatic Conversion During Analysis

Add the `--create-kml` flag when running an analysis:

```bash
python run.py analyze --bounds "-67.5,-10.1,-67.4,-10.0" --create-kml
```

This will automatically generate a KML file alongside the GeoJSON file when the analysis completes.

### 2. Converting Existing Files

Use the standalone conversion script to convert existing GeoJSON files:

```bash
# Convert a single file
python geojson_to_kml.py file path/to/your/file.geojson

# Convert all files in a directory
python geojson_to_kml.py directory path/to/your/directory

# Convert all files in a directory and its subdirectories
python geojson_to_kml.py directory path/to/your/directory --recursive
```

## Additional Options

### Command-Line Options for Single File Conversion

```bash
python geojson_to_kml.py file your_file.geojson --output output.kml --name "My Sites" --description "Archaeological sites in region X"
```

Options:
- `--output`: Path to save the output KML file (default: same name with .kml extension)
- `--name`: Name for the KML document
- `--description`: Description for the KML document

### Command-Line Options for Directory Conversion

```bash
python geojson_to_kml.py directory your_directory --output-directory output_directory --recursive
```

Options:
- `--output-directory`: Directory to save the output KML files (default: same as input directory)
- `--recursive`: Search for GeoJSON files in subdirectories

## Visualizing in Google Earth

After creating a KML file, you can visualize it in Google Earth:

1. Download and install [Google Earth Pro](https://www.google.com/earth/versions/) (desktop version)
2. Open Google Earth Pro
3. Go to File > Open
4. Select your KML file
5. The archaeological sites will appear as icons on the map
   - High confidence sites (≥0.7): Red stars
   - Medium confidence sites (0.4-0.7): Yellow stars
   - Low confidence sites (<0.4): White icons

## Programmatic Usage

You can also use the conversion functions in your own Python code:

```python
from src.utils.geo_converter import geojson_to_kml, convert_all_geojson_in_directory

# Convert a single file
kml_path = geojson_to_kml(
    "path/to/features.geojson",
    "path/to/output.kml",
    "Amazon Archaeological Sites",
    "Discovered using the Amazon Archaeological Discovery Tool"
)

# Convert all files in a directory
kml_paths = convert_all_geojson_in_directory(
    "path/to/directory",
    "path/to/output_directory",
    recursive=True
)
```

## Dependencies

The conversion tool has minimal dependencies:
- **Required**: lxml (for XML/KML creation)
- **Optional**: geopandas, fiona, shapely (for advanced geometry handling)

To install the required dependencies:

```bash
pip install lxml

# For advanced features (optional)
pip install -r requirements-geo.txt
```

## Troubleshooting

### Error: "lxml is required for KML conversion"

Install the lxml library:

```bash
pip install lxml
```

### GeoPandas or Fiona errors

If you see errors related to GeoPandas or Fiona, don't worry! The converter will automatically fall back to a simpler conversion method. However, for the best results, install the recommended dependencies:

```bash
pip install -r requirements-geo.txt
```

### KML Not Displaying Correctly in Google Earth

If your KML file doesn't display correctly in Google Earth:

1. Check that the GeoJSON file contains valid coordinates
2. Ensure the KML file is properly formatted XML (you can open it in a text editor)
3. Try converting with the `--name` and `--description` options to provide more context
4. Verify that Google Earth has internet access to load the icon images 