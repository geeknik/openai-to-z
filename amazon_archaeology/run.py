#!/usr/bin/env python3
"""
Main runner script for the Amazon Archaeological Discovery Tool.
This serves as the entry point to run the archaeological site detection pipeline.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import time

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import INITIAL_REGION_BOUNDS, DATA_DIR, SAMPLE_RESOLUTION, LIDAR_DATA_PATH
from src.preprocessing.data_fetcher import (
    get_data_sources_for_region,
    download_sample_lidar,
    fetch_sentinel_imagery,
    download_nasa_srtm,
    fetch_high_res_imagery_for_coordinate
)
from src.analysis.feature_detector import (
    detect_geometric_features,
    detect_vegetation_anomalies,
    merge_nearby_features,
    filter_by_confidence,
    save_features
)
from src.utils.cache import clear_cache
from src.utils.geo_converter import geojson_to_kml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(DATA_DIR / "amazon_archaeology.log")
    ]
)
logger = logging.getLogger(__name__)

# Pre-defined regions of archaeological interest in the Amazon
# Format: (name, description, bounds (west, south, east, north))
ARCHAEOLOGICAL_REGIONS = [
    (
        "upper-xingu",
        "Upper Xingu region with evidence of pre-Columbian settlements",
        (-54.0, -13.0, -53.0, -12.0)
    ),
    (
        "geoglyphs-acre", 
        "Geometric earthworks (geoglyphs) in Acre state",
        (-68.0, -10.2, -67.0, -9.5)  # Larger area covering multiple known geoglyphs
    ),
    (
        "monte-alegre",
        "Monte Alegre region with cave paintings and ancient settlements",
        (-54.5, -2.5, -53.5, -1.5)
    ),
    (
        "lower-amazon",
        "Lower Amazon with terra preta sites",
        (-55.0, -3.0, -54.0, -2.0)
    ),
    (
        "llanos-de-moxos",
        "Llanos de Moxos with extensive earthworks",
        (-66.0, -15.0, -65.0, -14.0)
    ),
    (
        "roosevelt-aripuana",
        "Roosevelt and Aripuanã rivers confluence with terra preta sites",
        (-60.5, -8.5, -59.5, -7.5)
    ),
    (
        "marajoara",
        "Marajó Island with Marajoara culture sites",
        (-50.0, -1.0, -49.0, 0.0)
    ),
    (
        "acre-small",
        "Specific area with known geometric geoglyphs in Acre",
        (-67.55, -10.05, -67.45, -9.95)  # Very small area for testing
    ),
    (
        "xapuri-acre",
        "Area with high concentration of geoglyphs near Xapuri in Acre state",
        (-68.53, -10.90, -68.40, -10.75)  # Known geoglyphs near Xapuri
    )
]

def parse_bounds(bounds_str: str) -> Tuple[float, float, float, float]:
    """Parse region bounds from string in format 'west,south,east,north'"""
    parts = bounds_str.split(",")
    if len(parts) != 4:
        raise ValueError("Bounds must be in format 'west,south,east,north'")
    
    return tuple(float(p) for p in parts)

def list_archaeological_regions():
    """Display available pre-defined archaeological regions"""
    print("\nPre-defined Archaeological Regions:")
    print("-" * 80)
    print(f"{'ID':<3} {'Name':<20} {'Description':<50} {'Bounds':<20}")
    print("-" * 80)
    
    for i, (name, desc, bounds) in enumerate(ARCHAEOLOGICAL_REGIONS):
        bounds_str = f"({bounds[0]},{bounds[1]})-({bounds[2]},{bounds[3]})"
        print(f"{i+1:<3} {name:<20} {desc:<50} {bounds_str:<20}")
    
    print("-" * 80)
    print("To use: python run.py --region 2  # To analyze the geoglyphs in Acre state")
    print("        python run.py --region geoglyphs-acre  # Same region, using name")
    print("-" * 80)

def list_amazon_lidar_datasets():
    """Display available Amazon LiDAR datasets"""
    try:
        from src.preprocessing.amazon_lidar import AMAZON_LIDAR_SOURCES
        
        print("\nSpecialized Amazon LiDAR Datasets:")
        print("-" * 100)
        print(f"{'Dataset ID':<25} {'Name':<35} {'Description':<50}")
        print("-" * 100)
        
        for source in AMAZON_LIDAR_SOURCES:
            # Truncate long descriptions
            description = source['description']
            if len(description) > 47:
                description = description[:44] + "..."
                
            print(f"{source['id']:<25} {source['name']:<35} {description:<50}")
            
            # Print regions within each dataset
            for region in source['regions']:
                bounds_str = f"({region['bounds'][0]:.1f},{region['bounds'][1]:.1f})-({region['bounds'][2]:.1f},{region['bounds'][3]:.1f})"
                print(f"  - Region: {region['name']:<15} {bounds_str}")
            print()
        
        print("-" * 100)
        print("To use: python run.py analyze --amazon-lidar ornl_slb_2008_2018 --bounds=-55.0,-5.0,-54.0,-4.0")
        print("        python run.py analyze --amazon-lidar ornl_manaus_2008 --amazon-region 'K34 Tower'")
        print("-" * 100)
    except ImportError:
        print("\nAmazon LiDAR module not available")

def create_example_data(example_dir: Path) -> Tuple[Path, Path]:
    """Create synthetic example data for testing"""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    
    # Create directory if it doesn't exist
    example_dir.mkdir(exist_ok=True, parents=True)
    
    lidar_path = example_dir / "lidar_sample.tif"
    satellite_path = example_dir / "satellite_sample.tif"
    
    # Create synthetic LiDAR data
    if not lidar_path.exists():
        # Create a 100x100 array with multiple geometric features
        data = np.zeros((100, 100), dtype=np.float32)
        
        # Add a clear rectangular feature
        data[40:60, 30:70] = 5.0
        
        # Add a circular feature
        center_y, center_x = 30, 75
        radius = 10
        y, x = np.ogrid[:100, :100]
        dist_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        circle_mask = dist_from_center <= radius
        data[circle_mask] = 4.0
        
        # Add a linear feature (like a wall)
        data[70:75, 20:80] = 6.0
        
        # Add minor noise (keep it small to ensure features are detectable)
        data += np.random.normal(0, 0.2, data.shape)
        
        # Create transform (each pixel is 10m)
        transform = from_origin(-64.0, -3.0, 0.001, 0.001)
        
        with rasterio.open(
            lidar_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs='+proj=latlong',
            transform=transform,
        ) as dst:
            dst.write(data, 1)
        logger.info(f"Created synthetic LiDAR data at {lidar_path}")
    
    # Create synthetic satellite data
    if not satellite_path.exists():
        # Create a 100x100 array with 4 bands (B, G, R, NIR)
        data = np.zeros((4, 100, 100), dtype=np.float32)
        
        # Set background values representing healthy vegetation
        data[0, :, :] = 0.1  # Blue
        data[1, :, :] = 0.15  # Green
        data[2, :, :] = 0.1  # Red
        data[3, :, :] = 0.4  # NIR (high NIR reflectance for vegetation)
        
        # Add a rectangular vegetation anomaly (archaeological site)
        data[0, 35:65, 25:75] = 0.15  # Blue
        data[1, 35:65, 25:75] = 0.25  # Green
        data[2, 35:65, 25:75] = 0.3  # Red (higher red)
        data[3, 35:65, 25:75] = 0.2  # NIR (lower NIR - stressed vegetation)
        
        # Add a circular vegetation anomaly (another site)
        center_y, center_x = 70, 30
        radius = 15
        y, x = np.ogrid[:100, :100]
        dist_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        circle_mask = dist_from_center <= radius
        
        # Create NDVI anomaly in circular area
        data[2, circle_mask] = 0.25  # Higher red
        data[3, circle_mask] = 0.25  # Lower NIR
        
        # Add linear feature (possible ancient path/road)
        mask_line = np.zeros((100, 100), dtype=bool)
        mask_line[45:48, 60:90] = True
        data[2, mask_line] = 0.22  # Distinct red
        data[3, mask_line] = 0.3  # Distinct NIR
        
        # Add minor noise
        data += np.random.normal(0, 0.01, data.shape)
        data = np.clip(data, 0, 1)
        
        # Create transform (each pixel is 10m)
        transform = from_origin(-64.0, -3.0, 0.001, 0.001)
        
        with rasterio.open(
            satellite_path,
            'w',
            driver='GTiff',
            height=data.shape[1],
            width=data.shape[2],
            count=4,
            dtype=data.dtype,
            crs='+proj=latlong',
            transform=transform,
        ) as dst:
            dst.write(data)
        logger.info(f"Created synthetic satellite data at {satellite_path}")
    
    return lidar_path, satellite_path

def verify_coordinate(lat: float, lon: float, radius: int = 500):
    """Fetch high-resolution imagery for verification of a specific coordinate"""
    logger.info(f"Fetching high-resolution imagery for verification at {lat}, {lon} with {radius}m radius")
    
    # Try to fetch high-resolution imagery
    imagery_path = fetch_high_res_imagery_for_coordinate(
        lat=lat,
        lon=lon,
        radius_meters=radius
    )
    
    if imagery_path:
        logger.info(f"Successfully downloaded high-resolution imagery to {imagery_path}")
        print(f"\n✅ High-resolution imagery saved to: {imagery_path}")
        print(f"Use GIS software like QGIS to view this imagery and verify potential archaeological site.")
    else:
        logger.error("Failed to download high-resolution imagery for verification")
        print("\n❌ Failed to download high-resolution imagery for verification.")
        print("Try a different coordinate or adjust the radius.")

def run_analysis(args):
    """Run the archaeological site analysis pipeline"""
    # Clear cache if requested
    if args.clear_cache:
        count = clear_cache()
        logger.info(f"Cleared {count} cached items")
    
    # Run in API mode if requested
    if args.api_mode:
        logger.info("Starting API server")
        import uvicorn
        from src.api.main import app
        uvicorn.run(app, host="0.0.0.0", port=8000)
        return
    
    # Get data paths
    lidar_path = None
    satellite_path = None
    
    if args.use_example_data:
        # Use example data
        logger.info("Using example data")
        example_dir = DATA_DIR / "example"
        lidar_path, satellite_path = create_example_data(example_dir)
        
        # Use example region bounds
        bounds = (-64.1, -3.1, -63.9, -2.9)
        logger.info(f"Using example region bounds: {bounds}")
        
    else:
        # Determine region bounds from arguments
        if args.region:
            # User specified a pre-defined region
            region_found = False
            
            # Check if numeric ID
            if args.region.isdigit():
                region_id = int(args.region)
                if 1 <= region_id <= len(ARCHAEOLOGICAL_REGIONS):
                    name, desc, bounds = ARCHAEOLOGICAL_REGIONS[region_id-1]
                    region_found = True
            else:
                # Check by name
                for name, desc, region_bounds in ARCHAEOLOGICAL_REGIONS:
                    if args.region.lower() == name.lower():
                        bounds = region_bounds
                        region_found = True
                        break
            
            if not region_found:
                logger.error(f"Region '{args.region}' not found. Use --list-regions to see available options.")
                sys.exit(1)
                
            logger.info(f"Using pre-defined region: {name} ({desc})")
            logger.info(f"Region bounds: {bounds}")
            
        elif args.bounds:
            # User specified custom bounds
            try:
                bounds = parse_bounds(args.bounds)
            except ValueError as e:
                logger.error(f"Invalid bounds format: {e}")
                sys.exit(1)
                
            logger.info(f"Using user-specified bounds: {bounds}")
            
        else:
            # No region or bounds specified, use default
            bounds = INITIAL_REGION_BOUNDS
            logger.info(f"Using default region bounds: {bounds}")
        
        logger.info(f"Running discovery pipeline for region: {bounds}")
        
        # Get available data sources
        sources = get_data_sources_for_region(bounds)
        logger.info(f"Data sources available: LiDAR={sources['lidar']['available']}, " +
                   f"Sentinel={sources['sentinel']['available']}, SRTM={sources['srtm']['available']}, " +
                   f"Amazon LiDAR={sources['amazon_lidar']['available']}")
        
        # Log information about Amazon-specific LiDAR datasets if available
        if sources['amazon_lidar']['available']:
            logger.info(f"Found {sources['amazon_lidar']['count']} specialized Amazon LiDAR datasets for this region")
            for idx, source in enumerate(sources['amazon_lidar']['sources']):
                logger.info(f"Amazon Dataset {idx+1}: {source['name']} - {source['url']}")
        
        # Download sample data
        # Check if a specific Amazon LiDAR dataset was requested
        if args.amazon_lidar and sources["amazon_lidar"]["available"]:
            amazon_datasets = [s for s in sources["amazon_lidar"]["sources"] if s["id"] == args.amazon_lidar]
            if amazon_datasets:
                logger.info(f"Using specified Amazon LiDAR dataset: {args.amazon_lidar}")
                try:
                    from src.preprocessing.amazon_lidar import download_amazon_lidar_sample
                    lidar_path = download_amazon_lidar_sample(
                        args.amazon_lidar,
                        bounds,
                        None,
                        args.resolution,
                        args.amazon_region
                    )
                    if lidar_path:
                        if lidar_path.suffix == '.tif':
                            logger.info(f"Successfully downloaded Amazon LiDAR data to {lidar_path}")
                            print(f"\n✅ Amazon LiDAR Dataset: {args.amazon_lidar}")
                            print(f"Data successfully downloaded and processed for analysis.")
                        elif lidar_path.suffix == '.guidance.txt':
                            logger.info(f"Amazon LiDAR guidance created at: {lidar_path}")
                            print(f"\n📘 Amazon LiDAR Dataset Information: {args.amazon_lidar}")
                            print("Automatic download was not possible. Please follow the instructions in the guidance file.")
                            
                            # Print the contents of the guidance file for user convenience
                            try:
                                with open(lidar_path, 'r') as f:
                                    guidance_content = f.read()
                                    print("\n" + "-"*80)
                                    print(guidance_content)
                                    print("-"*80 + "\n")
                            except Exception as e:
                                logger.error(f"Error reading guidance file: {e}")
                            
                            # Inform the user that we're falling back to SRTM data
                            print("\n⚠️ NOTE: We're proceeding with SRTM elevation data as a fallback.")
                            print("To use the specialized Amazon LiDAR data, you need to download it manually")
                            print("following the instructions in the guidance file above.\n")
                            
                            # Fall back to standard sources for analysis
                            lidar_path = None
                    else:
                        # No data found and no guidance file created
                        logger.warning(f"No Amazon LiDAR data available for dataset '{args.amazon_lidar}' in region {bounds}")
                        print(f"\n⚠️ No Amazon LiDAR data available for dataset '{args.amazon_lidar}' in this region.")
                        
                        # Fall back to standard sources
                        lidar_path = None
                except ImportError:
                    logger.error("Amazon LiDAR module not available")
            else:
                logger.warning(f"Specified Amazon LiDAR dataset '{args.amazon_lidar}' not found in available sources")
        
        # If we don't have a lidar path yet, try standard sources
        if not lidar_path and sources["lidar"]["available"]:
            # Download LiDAR data for the first available dataset
            dataset_id = sources["lidar"]["sources"][0].get("datasetId", "default")
            logger.info(f"Downloading LiDAR data from dataset: {dataset_id}")
            lidar_path = download_sample_lidar(dataset_id, bounds)
        elif not lidar_path:
            # Fallback to SRTM if available
            if sources["srtm"]["available"]:
                logger.info("No LiDAR data available, falling back to SRTM")
                srtm_filename = f"lidar_srtm_{bounds[0]}_{bounds[1]}_{bounds[2]}_{bounds[3]}_{args.resolution}m.tif"
                srtm_path = LIDAR_DATA_PATH / srtm_filename
                lidar_path = download_nasa_srtm(bounds, srtm_path, args.resolution)
                
                if not lidar_path or not os.path.exists(lidar_path):
                    logger.error("Failed to download SRTM elevation data")
                    lidar_path = None
                else:
                    logger.info(f"Successfully downloaded elevation data to {lidar_path}")
            else:
                logger.error("No elevation data (LiDAR or SRTM) available for this region")
                if not args.satellite_only:
                    sys.exit(1)
        
        # Download Sentinel imagery if available
        if sources["sentinel"]["available"]:
            logger.info("Downloading Sentinel imagery")
            satellite_path = fetch_sentinel_imagery(bounds)
            
            # Check if download succeeded
            if not satellite_path or not os.path.exists(satellite_path):
                logger.warning("Failed to download Sentinel imagery, continuing with elevation data only")
                satellite_path = None
            else:
                logger.info(f"Successfully downloaded satellite imagery to {satellite_path}")
        else:
            logger.warning("No Sentinel imagery available for this region")
            satellite_path = None

    # Ensure we have at least some data to proceed with
    if not lidar_path and not satellite_path:
        logger.error("Failed to obtain any usable data. Cannot proceed.")
        sys.exit(1)
    elif not lidar_path and not args.satellite_only:
        logger.error("Failed to obtain elevation data and --satellite-only not specified. Cannot proceed.")
        sys.exit(1)
    elif not lidar_path:
        logger.warning("No elevation data available, proceeding with satellite data only")
        lidar_features = []
    else:
        # Detect features in LiDAR data
        logger.info(f"Detecting features in elevation data: {lidar_path}")
        lidar_features = detect_geometric_features(
            lidar_path,
            min_size=args.min_size,
            max_size=args.max_size,
            sensitivity=args.sensitivity
        )
        logger.info(f"Detected {len(lidar_features)} features in elevation data")
    
    # Detect features in Sentinel imagery
    satellite_features = []
    if satellite_path and os.path.exists(satellite_path):
        logger.info(f"Detecting features in satellite imagery: {satellite_path}")
        satellite_features = detect_vegetation_anomalies(
            satellite_path,
            min_size=args.min_size,
            max_size=args.max_size,
            sensitivity=args.sensitivity
        )
        logger.info(f"Detected {len(satellite_features)} features in satellite imagery")
    else:
        logger.warning("Skipping satellite imagery analysis (no data available)")
    
    # Combine all features
    all_features = lidar_features + satellite_features
    
    if not all_features:
        logger.warning("No features detected in either elevation or satellite data")
        print("\n⚠️ No potential archaeological features were detected in this region.")
        print("Try adjusting parameters or selecting a different region.")
        sys.exit(0)
    
    # Merge nearby features
    logger.info(f"Merging {len(all_features)} features within {args.merge_distance}m distance")
    merged_features = merge_nearby_features(all_features, args.merge_distance)
    logger.info(f"Merged into {len(merged_features)} potential archaeological sites")
    
    # Filter by confidence if requested
    if args.min_confidence > 0:
        merged_features = filter_by_confidence(
            merged_features, 
            min_confidence=args.min_confidence,
            max_features=args.max_features
        )
    
    # Save features
    output_path = args.output
    if output_path:
        output_path = Path(output_path)
    
    saved_path = save_features(merged_features, output_path)
    if saved_path:
        logger.info(f"Saved features to {saved_path}")
        
        # Create KML file if requested
        if args.create_kml and len(merged_features) > 0:
            try:
                kml_path = geojson_to_kml(saved_path)
                logger.info(f"Created KML file at {kml_path}")
                print(f"\n🌎 KML file created: {kml_path}")
                print("  You can open this file in Google Earth to visualize the archaeological sites.")
            except Exception as e:
                logger.error(f"Failed to create KML file: {e}")
                print("\n⚠️ Failed to create KML file. Make sure lxml is installed: pip install lxml")
    
    # Create visualizations if requested
    if args.visualize and len(merged_features) > 0:
        try:
            from src.visualization.map_creator import create_feature_map, create_feature_plots
            
            logger.info("Creating visualizations")
            # Create interactive map
            map_path = create_feature_map(
                merged_features,
                center=None,  # Auto-center on features
                include_heatmap=True
            )
            logger.info(f"Created interactive map at {map_path}")
            
            # Create summary plots
            plot_paths = create_feature_plots(merged_features)
            if plot_paths:
                logger.info(f"Created {len(plot_paths)} summary plots in {plot_paths[0].parent}")
                
            print(f"\nVisualizations created:")
            print(f"- Interactive map: {map_path}")
            print(f"- Summary plots: {plot_paths[0].parent if plot_paths else 'None'}")
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    # Print summary
    print("\n📌 ====== DISCOVERY SUMMARY ======")
    print(f"Region: {bounds}")
    print(f"Data sources: {'Elevation data ✓' if lidar_path else 'Elevation data ✗'}, "
          f"{'Satellite imagery ✓' if satellite_path else 'Satellite imagery ✗'}")
    print(f"Total potential archaeological sites: {len(merged_features)}")
    
    if len(merged_features) > 0:
        high_confidence = len([f for f in merged_features if f["confidence"] >= 0.7])
        medium_confidence = len([f for f in merged_features if 0.4 <= f["confidence"] < 0.7])
        low_confidence = len([f for f in merged_features if f["confidence"] < 0.4])
        
        print(f"Confidence levels:")
        print(f"- High confidence (>= 0.7): {high_confidence}")
        print(f"- Medium confidence (0.4-0.7): {medium_confidence}")
        print(f"- Low confidence (< 0.4): {low_confidence}")
        
        # Show top 5 sites sorted by confidence
        print("\nTop potential archaeological sites:")
        for i, feature in enumerate(sorted(merged_features, key=lambda f: f["confidence"], reverse=True)[:5]):
            print(f"{i+1}. Type: {feature['type']}, Confidence: {feature['confidence']:.2f}, " +
                  f"Coordinates: {feature['coordinates']['lon']:.5f}, {feature['coordinates']['lat']:.5f}, " +
                  f"Size: {feature['size']['area_m2']:.1f}m²")
    
    print("=================================\n")
    if saved_path:
        print(f"📄 Full results saved to: {saved_path}")
    
    # For real-world data, provide further analysis suggestions
    if not args.use_example_data:
        print("\n📋 Next steps:")
        print("1. Import these coordinates into QGIS or Google Earth for further analysis")
        print("   (use --create-kml to generate Google Earth compatible files)")
        print("2. Try analyzing neighboring regions to expand the search area")
        print("3. Run with --visualize --sensitivity 0.7 for more precise detection")
        print("4. Use --region parameter to explore other archaeological hotspots")
        print("   (python run.py list-regions to see options)")
        print("5. Explore specialized Amazon LiDAR datasets:")
        print("   (python run.py list-amazon-datasets to see options)")
        print("   python run.py analyze --amazon-lidar ornl_slb_2008_2018 --bounds=...")
        print("6. Convert existing GeoJSON files to KML: python geojson_to_kml.py file path/to/file.geojson")
    
    return merged_features

def main():
    """Main function to run the archaeological discovery pipeline."""
    parser = argparse.ArgumentParser(description="Amazon Archaeological Discovery Tool")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Analysis command (the original functionality)
    analysis_parser = subparsers.add_parser("analyze", help="Run archaeological site detection")
    
    # Define command line arguments for analysis
    analysis_parser.add_argument(
        "--bounds", 
        help="Region bounds in format 'west,south,east,north'"
    )
    analysis_parser.add_argument(
        "--region",
        help="Use a pre-defined archaeological region by ID or name"
    )
    analysis_parser.add_argument(
        "--list-regions",
        action="store_true",
        help="List pre-defined archaeological regions"
    )
    analysis_parser.add_argument(
        "--min-size", 
        type=int, 
        default=80,
        help="Minimum feature size in meters"
    )
    analysis_parser.add_argument(
        "--max-size", 
        type=int, 
        default=500,
        help="Maximum feature size in meters"
    )
    analysis_parser.add_argument(
        "--merge-distance", 
        type=float, 
        default=100.0,
        help="Distance in meters to merge nearby features"
    )
    analysis_parser.add_argument(
        "--sensitivity", 
        type=float, 
        default=0.5,
        help="Detection sensitivity (0-1)"
    )
    analysis_parser.add_argument(
        "--output", 
        help="Output path for detected features (GeoJSON)"
    )
    analysis_parser.add_argument(
        "--clear-cache", 
        action="store_true",
        help="Clear cache before running"
    )
    analysis_parser.add_argument(
        "--api-mode", 
        action="store_true",
        help="Run in API server mode"
    )
    analysis_parser.add_argument(
        "--use-example-data", 
        action="store_true",
        help="Use example data instead of fetching new data"
    )
    analysis_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization of results"
    )
    analysis_parser.add_argument(
        "--satellite-only",
        action="store_true",
        help="Run analysis with only satellite data if elevation data is unavailable"
    )
    analysis_parser.add_argument(
        "--resolution", 
        type=int, 
        default=SAMPLE_RESOLUTION,
        help="Resolution in meters for downloaded data"
    )
    analysis_parser.add_argument(
        "--min-confidence", 
        type=float, 
        default=0.0,
        help="Minimum confidence level for features (0-1)"
    )
    analysis_parser.add_argument(
        "--max-features", 
        type=int, 
        default=0,
        help="Maximum number of features to include (0 for all)"
    )
    analysis_parser.add_argument(
        "--create-kml",
        action="store_true",
        help="Create a KML file for Google Earth visualization"
    )
    analysis_parser.add_argument(
        "--amazon-lidar",
        help="Use a specific Amazon LiDAR dataset (e.g., ornl_slb_2008_2018, ornl_manaus_2008)"
    )
    analysis_parser.add_argument(
        "--amazon-region",
        help="Specify a particular region within the Amazon LiDAR dataset"
    )
    
    # Verification command
    verify_parser = subparsers.add_parser("verify", help="Fetch high-resolution imagery for verification")
    verify_parser.add_argument(
        "--lat",
        type=float,
        required=True,
        help="Latitude of the site to verify"
    )
    verify_parser.add_argument(
        "--lon",
        type=float,
        required=True,
        help="Longitude of the site to verify"
    )
    verify_parser.add_argument(
        "--radius",
        type=int,
        default=500,
        help="Radius in meters around the coordinate (default: 500)"
    )
    
    # List regions command
    list_parser = subparsers.add_parser("list-regions", help="List pre-defined archaeological regions")
    
    # List Amazon LiDAR datasets command
    list_amazon_parser = subparsers.add_parser("list-amazon-datasets", help="List specialized Amazon LiDAR datasets")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle different commands
    if not args.command:
        # For backward compatibility, support the original format without subcommands
        if hasattr(args, 'list_regions') and args.list_regions:
            list_archaeological_regions()
            return
        elif hasattr(args, 'api_mode') and args.api_mode:
            logger.info("Starting API server")
            import uvicorn
            from src.api.main import app
            uvicorn.run(app, host="0.0.0.0", port=8000)
            return
        else:
            # Default to analysis with original arguments
            run_analysis(args)
    elif args.command == "list-regions":
        list_archaeological_regions()
        return
    elif args.command == "list-amazon-datasets":
        list_amazon_lidar_datasets()
        return
    elif args.command == "verify":
        verify_coordinate(args.lat, args.lon, args.radius)
        return
    elif args.command == "analyze":
        run_analysis(args)
        return

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1) 