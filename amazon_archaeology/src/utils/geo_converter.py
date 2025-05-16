"""
Geospatial format conversion utilities for the Amazon Archaeological Discovery Tool.
Primarily used for converting GeoJSON outputs to other formats like KML for visualization.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import tempfile

try:
    from lxml import etree
except ImportError:
    logging.warning("lxml not installed. Install with: pip install lxml")
    etree = None

# Try to import geopandas and check if it actually works
GEOPANDAS_AVAILABLE = False
try:
    import geopandas as gpd
    import fiona
    from shapely.geometry import Point, Polygon, LineString
    
    # Modified check for fiona - don't check for path attribute as it's not reliable on macOS
    # Check for basic functionality instead
    try:
        # Create a simple test geodataframe to verify functionality
        test_df = gpd.GeoDataFrame(
            {"col1": [1]}, 
            geometry=[Point(0, 0)]
        )
        # Try a simple save/load operation to verify fiona driver functionality
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "test.geojson"
            test_df.to_file(tmp_path, driver="GeoJSON")
            test_load = gpd.read_file(tmp_path)
            assert len(test_load) == 1  # Verify data was saved and loaded correctly
        
        GEOPANDAS_AVAILABLE = True
        logging.info("GeoPandas is available and working correctly")
    except Exception as e:
        raise RuntimeError(f"GeoPandas/Fiona functionality test failed: {e}")
        
except (ImportError, RuntimeError) as e:
    logging.warning(f"GeoPandas not available or not fully functional: {e}")
    logging.warning("Using simple JSON parsing for GeoJSON conversion")

# Setup logging
logger = logging.getLogger(__name__)

def geojson_to_kml(
    geojson_path: Union[str, Path],
    kml_path: Optional[Union[str, Path]] = None,
    name: str = "Amazon Archaeological Sites",
    description: str = "Potential archaeological sites detected by the Amazon Archaeological Discovery Tool"
) -> Path:
    """
    Convert GeoJSON file to KML format for visualization in Google Earth.
    
    Args:
        geojson_path: Path to the input GeoJSON file
        kml_path: Path to save the KML file (None for automatic generation)
        name: Name for the KML document
        description: Description for the KML document
        
    Returns:
        Path to the saved KML file
        
    Raises:
        FileNotFoundError: If the input GeoJSON file doesn't exist
        ValueError: If the input GeoJSON is invalid or dependencies are missing
    """
    # Check required dependencies
    if etree is None:
        raise ImportError("lxml is required for KML conversion. Install with: pip install lxml")
    
    # Convert to Path objects
    geojson_path = Path(geojson_path)
    
    # Generate output path if not provided
    if kml_path is None:
        kml_path = geojson_path.with_suffix('.kml')
    else:
        kml_path = Path(kml_path)
    
    # Ensure the input file exists
    if not geojson_path.exists():
        raise FileNotFoundError(f"Input GeoJSON file not found: {geojson_path}")
    
    try:
        # Create KML document structure
        kml = etree.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
        document = etree.SubElement(kml, "Document")
        
        # Add document name and description
        etree.SubElement(document, "name").text = name
        etree.SubElement(document, "description").text = description
        
        # Define styles for different confidence levels
        styles = {
            "high": {
                "id": "highConfidenceStyle",
                "icon": "http://maps.google.com/mapfiles/kml/paddle/red-stars.png",
                "color": "ff0000ff",  # AABBGGRR format
                "scale": 1.2
            },
            "medium": {
                "id": "mediumConfidenceStyle",
                "icon": "http://maps.google.com/mapfiles/kml/paddle/ylw-stars.png",
                "color": "ff00ffff",  # AABBGGRR format
                "scale": 1.0
            },
            "low": {
                "id": "lowConfidenceStyle",
                "icon": "http://maps.google.com/mapfiles/kml/paddle/wht-blank.png",
                "color": "ffffffff",  # AABBGGRR format
                "scale": 0.8
            }
        }
        
        # Add styles to document
        for style_id, style_props in styles.items():
            style = etree.SubElement(document, "Style", id=style_props["id"])
            
            # Icon style
            icon_style = etree.SubElement(style, "IconStyle")
            etree.SubElement(icon_style, "color").text = style_props["color"]
            etree.SubElement(icon_style, "scale").text = str(style_props["scale"])
            icon = etree.SubElement(icon_style, "Icon")
            etree.SubElement(icon, "href").text = style_props["icon"]
            
            # Label style
            label_style = etree.SubElement(style, "LabelStyle")
            etree.SubElement(label_style, "scale").text = str(style_props["scale"])
            
            # Line style for polygons
            line_style = etree.SubElement(style, "LineStyle")
            etree.SubElement(line_style, "color").text = style_props["color"]
            etree.SubElement(line_style, "width").text = "2"
            
            # Polygon style
            poly_style = etree.SubElement(style, "PolyStyle")
            etree.SubElement(poly_style, "color").text = style_props["color"].replace("ff", "80")  # 50% opacity
            etree.SubElement(poly_style, "outline").text = "1"
            etree.SubElement(poly_style, "fill").text = "1"
        
        # Choose conversion method based on available dependencies
        if GEOPANDAS_AVAILABLE:
            # Use GeoPandas for advanced conversion
            logger.info("Using GeoPandas for GeoJSON to KML conversion")
            _convert_with_geopandas(geojson_path, document, styles)
        else:
            # Use simple JSON parsing
            logger.info("Using direct JSON parsing for GeoJSON to KML conversion")
            _convert_with_json(geojson_path, document, styles)
        
        # Create the KML file
        tree = etree.ElementTree(kml)
        tree.write(kml_path, pretty_print=True, xml_declaration=True, encoding='utf-8')
        
        logger.info(f"Successfully converted {geojson_path} to KML format at {kml_path}")
        return kml_path
    
    except Exception as e:
        logger.error(f"Error converting GeoJSON to KML: {e}")
        raise ValueError(f"Failed to convert GeoJSON to KML: {str(e)}")

def _convert_with_geopandas(geojson_path: Path, document, styles):
    """Convert GeoJSON to KML using GeoPandas"""
    # Read the GeoJSON file using geopandas
    gdf = gpd.read_file(geojson_path)
    
    # Convert each feature to KML
    for idx, feature in gdf.iterrows():
        # Create placemark
        placemark = etree.SubElement(document, "Placemark")
        
        # Add name and description
        feature_type = feature.get("type", "Unknown")
        confidence = feature.get("confidence", 0)
        
        # Create name based on type and confidence
        name_text = f"{feature_type.replace('_', ' ').title()} ({confidence:.2f})"
        etree.SubElement(placemark, "name").text = name_text
        
        # Create detailed description from properties
        description_text = "<![CDATA["
        description_text += f"<h3>{name_text}</h3>"
        description_text += "<table border='1'>"
        
        # Add coordinates
        if "coordinates" in feature and isinstance(feature["coordinates"], dict):
            coords = feature["coordinates"]
            lat = coords.get("lat", "Unknown")
            lon = coords.get("lon", "Unknown")
            description_text += f"<tr><td>Coordinates</td><td>{lat}, {lon}</td></tr>"
        
        # Add size information
        if "size" in feature and isinstance(feature["size"], dict):
            size = feature["size"]
            width = size.get("width_m", "Unknown")
            height = size.get("height_m", "Unknown")
            area = size.get("area_m2", "Unknown")
            description_text += f"<tr><td>Size</td><td>Width: {width}m, Height: {height}m, Area: {area}m²</td></tr>"
        
        # Add shape information if available
        if "shape" in feature:
            description_text += f"<tr><td>Shape</td><td>{feature['shape']}</td></tr>"
        
        # Add confidence information
        description_text += f"<tr><td>Confidence</td><td>{confidence:.2f}</td></tr>"
        
        # Add metadata if available
        if "metadata" in feature and isinstance(feature["metadata"], dict):
            description_text += "<tr><td colspan='2'><b>Metadata</b></td></tr>"
            for key, value in feature["metadata"].items():
                description_text += f"<tr><td>{key}</td><td>{value}</td></tr>"
        
        description_text += "</table>"
        description_text += "]]>"
        
        etree.SubElement(placemark, "description").text = description_text
        
        # Set style based on confidence level
        if confidence >= 0.7:
            etree.SubElement(placemark, "styleUrl").text = f"#{styles['high']['id']}"
        elif confidence >= 0.4:
            etree.SubElement(placemark, "styleUrl").text = f"#{styles['medium']['id']}"
        else:
            etree.SubElement(placemark, "styleUrl").text = f"#{styles['low']['id']}"
        
        # Add geometry
        geom = feature.geometry
        
        # Add Point geometry
        if isinstance(geom, Point):
            point = etree.SubElement(placemark, "Point")
            coords_text = f"{geom.x},{geom.y},0"
            etree.SubElement(point, "coordinates").text = coords_text
        
        # Add LineString geometry
        elif isinstance(geom, LineString):
            line = etree.SubElement(placemark, "LineString")
            coords_text = " ".join([f"{x},{y},0" for x, y in geom.coords])
            etree.SubElement(line, "coordinates").text = coords_text
        
        # Add Polygon geometry
        elif isinstance(geom, Polygon):
            polygon = etree.SubElement(placemark, "Polygon")
            outer = etree.SubElement(polygon, "outerBoundaryIs")
            ring = etree.SubElement(outer, "LinearRing")
            coords_text = " ".join([f"{x},{y},0" for x, y in geom.exterior.coords])
            etree.SubElement(ring, "coordinates").text = coords_text
            
            # Add inner rings if any
            for inner_ring in geom.interiors:
                inner = etree.SubElement(polygon, "innerBoundaryIs")
                inner_linear_ring = etree.SubElement(inner, "LinearRing")
                inner_coords_text = " ".join([f"{x},{y},0" for x, y in inner_ring.coords])
                etree.SubElement(inner_linear_ring, "coordinates").text = inner_coords_text

def _convert_with_json(geojson_path: Path, document, styles):
    """Convert GeoJSON to KML using direct JSON parsing"""
    # Read the GeoJSON file using json
    with open(geojson_path) as f:
        geojson_data = json.load(f)
    
    # Process GeoJSON features
    features = []
    if "type" in geojson_data:
        if geojson_data["type"] == "FeatureCollection" and "features" in geojson_data:
            features = geojson_data["features"]
        elif geojson_data["type"] == "Feature":
            features = [geojson_data]
    
    # Convert each feature to KML
    for feature in features:
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        
        # Skip features without geometry
        if not geometry:
            continue
        
        # Create placemark
        placemark = etree.SubElement(document, "Placemark")
        
        # Add name and description
        feature_type = properties.get("type", "Unknown")
        confidence = properties.get("confidence", 0)
        
        # Create name based on type and confidence
        name_text = f"{feature_type.replace('_', ' ').title()} ({confidence:.2f})"
        etree.SubElement(placemark, "name").text = name_text
        
        # Create detailed description from properties
        description_text = "<![CDATA["
        description_text += f"<h3>{name_text}</h3>"
        description_text += "<table border='1'>"
        
        # Add coordinates
        if "coordinates" in properties and isinstance(properties["coordinates"], dict):
            coords = properties["coordinates"]
            lat = coords.get("lat", "Unknown")
            lon = coords.get("lon", "Unknown")
            description_text += f"<tr><td>Coordinates</td><td>{lat}, {lon}</td></tr>"
        
        # Add size information
        if "size" in properties and isinstance(properties["size"], dict):
            size = properties["size"]
            width = size.get("width_m", "Unknown")
            height = size.get("height_m", "Unknown")
            area = size.get("area_m2", "Unknown")
            description_text += f"<tr><td>Size</td><td>Width: {width}m, Height: {height}m, Area: {area}m²</td></tr>"
        
        # Add shape information if available
        if "shape" in properties:
            description_text += f"<tr><td>Shape</td><td>{properties['shape']}</td></tr>"
        
        # Add confidence information
        description_text += f"<tr><td>Confidence</td><td>{confidence:.2f}</td></tr>"
        
        # Add metadata if available
        if "metadata" in properties and isinstance(properties["metadata"], dict):
            description_text += "<tr><td colspan='2'><b>Metadata</b></td></tr>"
            for key, value in properties["metadata"].items():
                description_text += f"<tr><td>{key}</td><td>{value}</td></tr>"
        
        description_text += "</table>"
        description_text += "]]>"
        
        etree.SubElement(placemark, "description").text = description_text
        
        # Set style based on confidence level
        if confidence >= 0.7:
            etree.SubElement(placemark, "styleUrl").text = f"#{styles['high']['id']}"
        elif confidence >= 0.4:
            etree.SubElement(placemark, "styleUrl").text = f"#{styles['medium']['id']}"
        else:
            etree.SubElement(placemark, "styleUrl").text = f"#{styles['low']['id']}"
        
        # Add geometry based on type
        geom_type = geometry.get("type", "").lower()
        coordinates = geometry.get("coordinates", [])
        
        # Point geometry
        if geom_type == "point" and len(coordinates) >= 2:
            point = etree.SubElement(placemark, "Point")
            # GeoJSON coordinates are [longitude, latitude]
            coords_text = f"{coordinates[0]},{coordinates[1]},0"
            etree.SubElement(point, "coordinates").text = coords_text
        
        # LineString geometry
        elif geom_type == "linestring" and coordinates:
            line = etree.SubElement(placemark, "LineString")
            coords_text = " ".join([f"{coord[0]},{coord[1]},0" for coord in coordinates])
            etree.SubElement(line, "coordinates").text = coords_text
        
        # Polygon geometry
        elif geom_type == "polygon" and coordinates:
            polygon = etree.SubElement(placemark, "Polygon")
            
            # Outer ring
            if coordinates:
                outer = etree.SubElement(polygon, "outerBoundaryIs")
                ring = etree.SubElement(outer, "LinearRing")
                coords_text = " ".join([f"{coord[0]},{coord[1]},0" for coord in coordinates[0]])
                etree.SubElement(ring, "coordinates").text = coords_text
            
            # Inner rings
            for i in range(1, len(coordinates)):
                inner = etree.SubElement(polygon, "innerBoundaryIs")
                inner_ring = etree.SubElement(inner, "LinearRing")
                inner_coords_text = " ".join([f"{coord[0]},{coord[1]},0" for coord in coordinates[i]])
                etree.SubElement(inner_ring, "coordinates").text = inner_coords_text

def convert_all_geojson_in_directory(
    directory: Union[str, Path],
    output_directory: Optional[Union[str, Path]] = None,
    recursive: bool = False
) -> List[Path]:
    """
    Convert all GeoJSON files in a directory to KML format.
    
    Args:
        directory: Input directory containing GeoJSON files
        output_directory: Directory to save KML files (None for same as input)
        recursive: Whether to search recursively in subdirectories
        
    Returns:
        List of paths to the converted KML files
    """
    directory = Path(directory)
    
    if output_directory is None:
        output_directory = directory
    else:
        output_directory = Path(output_directory)
        os.makedirs(output_directory, exist_ok=True)
    
    # Find all GeoJSON files
    if recursive:
        geojson_files = list(directory.glob("**/*.geojson"))
    else:
        geojson_files = list(directory.glob("*.geojson"))
    
    if not geojson_files:
        logger.warning(f"No GeoJSON files found in {directory}")
        return []
    
    # Convert each file
    converted_files = []
    for geojson_file in geojson_files:
        try:
            # Generate output path
            relative_path = geojson_file.relative_to(directory)
            output_path = output_directory / relative_path.with_suffix('.kml')
            
            # Create parent directories if needed
            os.makedirs(output_path.parent, exist_ok=True)
            
            # Convert the file
            kml_path = geojson_to_kml(geojson_file, output_path)
            converted_files.append(kml_path)
            
            logger.info(f"Converted {geojson_file} to {kml_path}")
        except Exception as e:
            logger.error(f"Error converting {geojson_file}: {e}")
    
    return converted_files 