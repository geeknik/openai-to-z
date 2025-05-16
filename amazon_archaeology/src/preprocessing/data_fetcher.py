"""
Data fetching utilities for LiDAR and satellite imagery.
Focuses on free, open-source data to minimize costs.
"""

import os
import requests
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import tempfile
import logging
import time
import math

import ee
import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from datetime import datetime, timedelta

from ..config import (
    GEE_SERVICE_ACCOUNT, 
    GEE_KEY_FILE,
    LIDAR_DATA_PATH,
    SATELLITE_DATA_PATH,
    SAMPLE_RESOLUTION,
    OPEN_TOPO_KEY
)
from ..utils.cache import cache_result

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Maximum region size for Earth Engine downloads (in degrees)
MAX_REGION_SIZE_DEGREES = 0.5
# Maximum tile size for downloads (in MB)
MAX_DOWNLOAD_SIZE_MB = 40

# Initialize Google Earth Engine
def initialize_gee():
    """Initialize Google Earth Engine with service account if available."""
    try:
        if GEE_SERVICE_ACCOUNT and GEE_KEY_FILE and os.path.exists(GEE_KEY_FILE):
            credentials = ee.ServiceAccountCredentials(GEE_SERVICE_ACCOUNT, GEE_KEY_FILE)
            ee.Initialize(credentials)
            logger.info(f"Google Earth Engine initialized with service account: {GEE_SERVICE_ACCOUNT}")
        else:
            # Try default authentication
            ee.Initialize()
            logger.info("Google Earth Engine initialized with default credentials")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Google Earth Engine: {e}")
        return False

# Attempt GEE initialization but don't crash if unavailable
GEE_AVAILABLE = initialize_gee()

@cache_result(expire_seconds=86400*30)  # Cache for 30 days
def fetch_lidar_metadata(
    bounds: Tuple[float, float, float, float],
    max_results: int = 20
) -> List[Dict]:
    """
    Fetch metadata for available open LiDAR datasets within bounds.
    Uses OpenTopography API which is free for research.
    
    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat)
        max_results: Maximum number of datasets to return
        
    Returns:
        List of metadata dictionaries for available LiDAR datasets
    """
    # OpenTopography API endpoint (free for research)
    api_url = "https://portal.opentopography.org/API/datasets"
    
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # Construct query parameters
    params = {
        "output": "json",
        "minLon": min_lon,
        "minLat": min_lat,
        "maxLon": max_lon,
        "maxLat": max_lat,
        "limit": max_results
    }
    
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        logger.info(f"Found {len(data.get('Items', []))} OpenTopography datasets for region")
        return data.get("Items", [])
    except requests.RequestException as e:
        logger.error(f"Error fetching LiDAR metadata: {e}")
        return []

def download_nasa_srtm(
    bounds: Tuple[float, float, float, float],
    output_path: Path,
    resolution: int = SAMPLE_RESOLUTION
) -> Optional[Path]:
    """
    Download SRTM data directly from NASA Earth Data.
    Uses the public SRTM data hosted by Amazon S3 without requiring GDAL tools.
    
    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat)
        output_path: Path to save the downloaded file
        resolution: Output resolution in meters
        
    Returns:
        Path to downloaded file or None if failed
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    
    try:
        # For SRTM, we need to identify the tiles that cover our region
        # SRTM tiles are 1x1 degree and named by their southwest corner
        # e.g., N00E104 is the tile from 0-1°N, 104-105°E
        
        # Calculate the southwest corners of all tiles covering our region
        min_lon_tile = math.floor(min_lon)
        min_lat_tile = math.floor(min_lat)
        max_lon_tile = math.floor(max_lon)
        max_lat_tile = math.floor(max_lat)
        
        # Create a list of all tiles we need
        tiles = []
        for lat in range(min_lat_tile, max_lat_tile + 1):
            for lon in range(min_lon_tile, max_lon_tile + 1):
                # SRTM tile naming convention
                ns = "N" if lat >= 0 else "S"
                ew = "E" if lon >= 0 else "W"
                
                lat_abs = abs(lat)
                lon_abs = abs(lon)
                
                tile_name = f"{ns}{lat_abs:02d}{ew}{lon_abs:03d}"
                tiles.append((tile_name, lat, lon))
        
        logger.info(f"Need to download {len(tiles)} SRTM tiles: {[t[0] for t in tiles]}")
        
        # If we have only one tile, we can download and use it directly
        if len(tiles) == 1:
            tile_name, lat, lon = tiles[0]
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            
            # URL for AWS Open Data SRTM
            url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{ns}{abs(lat):02d}/{tile_name}.hgt.gz"
            
            # Download and decompress tile
            try:
                # Create temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    # Download compressed file
                    logger.info(f"Downloading SRTM tile: {tile_name} from {url}")
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    # Save compressed tile
                    tile_gz_path = temp_path / f"{tile_name}.hgt.gz"
                    with open(tile_gz_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Decompress tile
                    import gzip
                    tile_path = temp_path / f"{tile_name}.hgt"
                    with gzip.open(tile_gz_path, 'rb') as f_in:
                        with open(tile_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Convert HGT to GeoTIFF using rasterio
                    # HGT files are 16-bit signed integers with 3601x3601 pixels (for SRTM 1 arc-second)
                    # They are big-endian with no header
                    import rasterio
                    from rasterio.transform import from_origin
                    
                    # Read data
                    with open(tile_path, 'rb') as f:
                        # SRTM data is 16-bit signed integers in big-endian format
                        # Some systems might have issues with the dtype('>i2') notation
                        # Use the more explicit np.dtype('int16').newbyteorder('>') instead
                        data = np.fromfile(f, dtype=np.dtype('int16').newbyteorder('>')).reshape((3601, 3601))
                    
                    # Create transform (SRTM tiles are 1-degree with southwest origin)
                    transform = from_origin(lon, lat + 1, 1/3600, 1/3600)  # 1 arcsecond resolution
                    
                    # Create new raster with cropped data
                    # Calculate array indices for the requested region
                    # SRTM has origin at top-left with y increasing downward
                    row_min = max(0, int((lat + 1 - max_lat) * 3600))
                    row_max = min(3601, int((lat + 1 - min_lat) * 3600))
                    col_min = max(0, int((min_lon - lon) * 3600))
                    col_max = min(3601, int((max_lon - lon) * 3600))
                    
                    # Crop data
                    cropped_data = data[row_min:row_max, col_min:col_max]
                    
                    # Calculate new transform for cropped data
                    new_west = lon + col_min/3600
                    new_north = lat + 1 - row_min/3600
                    new_transform = from_origin(new_west, new_north, 1/3600, 1/3600)
                    
                    # Write to GeoTIFF
                    with rasterio.open(
                        output_path,
                        'w',
                        driver='GTiff',
                        height=cropped_data.shape[0],
                        width=cropped_data.shape[1],
                        count=1,
                        dtype=cropped_data.dtype,
                        crs='+proj=latlong',
                        transform=new_transform,
                    ) as dst:
                        dst.write(cropped_data, 1)
                    
                    logger.info(f"SRTM data saved to {output_path}")
                    return output_path
            except Exception as e:
                logger.error(f"Error processing single SRTM tile: {e}")
                return None
        else:
            # Multiple tiles - we'll use a simpler approach without GDAL
            # For each tile, we'll download it and extract just the portion we need
            
            # Create a new array to hold our merged data
            # Determine dimensions of our output array (assuming 1 arcsecond resolution)
            width_pixels = int((max_lon - min_lon) * 3600) + 1
            height_pixels = int((max_lat - min_lat) * 3600) + 1
            merged_data = np.zeros((height_pixels, width_pixels), dtype=np.int16)
            
            try:
                # Create temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    # Download and process each tile
                    for tile_name, lat, lon in tiles:
                        ns = "N" if lat >= 0 else "S"
                        ew = "E" if lon >= 0 else "W"
                        
                        # URL for AWS Open Data SRTM
                        url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{ns}{abs(lat):02d}/{tile_name}.hgt.gz"
                        
                        try:
                            # Download compressed file
                            logger.info(f"Downloading SRTM tile: {tile_name} from {url}")
                            response = requests.get(url, stream=True)
                            response.raise_for_status()
                            
                            # Save compressed tile
                            tile_gz_path = temp_path / f"{tile_name}.hgt.gz"
                            with open(tile_gz_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            
                            # Decompress tile
                            import gzip
                            tile_path = temp_path / f"{tile_name}.hgt"
                            with gzip.open(tile_gz_path, 'rb') as f_in:
                                with open(tile_path, 'wb') as f_out:
                                    shutil.copyfileobj(f_in, f_out)
                            
                            # Read the HGT file
                            with open(tile_path, 'rb') as f:
                                tile_data = np.fromfile(f, dtype=np.dtype('int16').newbyteorder('>')).reshape((3601, 3601))
                            
                            # Calculate which part of this tile we need
                            # and where it should go in our merged data
                            
                            # Tile boundaries
                            tile_west = lon
                            tile_east = lon + 1
                            tile_south = lat
                            tile_north = lat + 1
                            
                            # Overlap with our region
                            overlap_west = max(min_lon, tile_west)
                            overlap_east = min(max_lon, tile_east)
                            overlap_south = max(min_lat, tile_south)
                            overlap_north = min(max_lat, tile_north)
                            
                            # Skip if no overlap
                            if overlap_west >= overlap_east or overlap_south >= overlap_north:
                                continue
                                
                            # Calculate pixel indices in tile
                            # SRTM has origin at top-left with y increasing downward
                            tile_row_min = int(round((tile_north - overlap_north) * 3600))
                            tile_row_max = int(round((tile_north - overlap_south) * 3600))
                            tile_col_min = int(round((overlap_west - tile_west) * 3600))
                            tile_col_max = int(round((overlap_east - tile_west) * 3600))
                            
                            # Calculate pixel indices in merged data
                            merged_row_min = int(round((max_lat - overlap_north) * 3600))
                            merged_row_max = int(round((max_lat - overlap_south) * 3600))
                            merged_col_min = int(round((overlap_west - min_lon) * 3600))
                            merged_col_max = int(round((overlap_east - min_lon) * 3600))
                            
                            # Copy data
                            merged_data[
                                merged_row_min:merged_row_max,
                                merged_col_min:merged_col_max
                            ] = tile_data[
                                tile_row_min:tile_row_max,
                                tile_col_min:tile_col_max
                            ]
                        except Exception as e:
                            logger.error(f"Error processing tile {tile_name}: {e}")
                            # Continue with other tiles
                    
                    # Create GeoTIFF with merged data
                    import rasterio
                    from rasterio.transform import from_origin
                    
                    # Create transform for merged data
                    transform = from_origin(min_lon, max_lat, 1/3600, 1/3600)
                    
                    # Subsample to requested resolution if needed
                    if resolution > 30:  # If requested resolution is coarser than 1 arcsecond (~30m)
                        scale_factor = int(resolution / 30)
                        
                        # Simple subsampling - take every nth pixel
                        # For production, consider using a proper resampling algorithm
                        subsampled_height = max(1, merged_data.shape[0] // scale_factor)
                        subsampled_width = max(1, merged_data.shape[1] // scale_factor)
                        subsampled_data = np.zeros((subsampled_height, subsampled_width), dtype=merged_data.dtype)
                        
                        for i in range(subsampled_height):
                            for j in range(subsampled_width):
                                subsampled_data[i, j] = merged_data[i * scale_factor, j * scale_factor]
                        
                        # Create new transform for subsampled data
                        subsampled_transform = from_origin(min_lon, max_lat, (1/3600) * scale_factor, (1/3600) * scale_factor)
                        
                        # Write subsampled data
                        with rasterio.open(
                            output_path,
                            'w',
                            driver='GTiff',
                            height=subsampled_data.shape[0],
                            width=subsampled_data.shape[1],
                            count=1,
                            dtype=subsampled_data.dtype,
                            crs='+proj=latlong',
                            transform=subsampled_transform,
                        ) as dst:
                            dst.write(subsampled_data, 1)
                    else:
                        # Write full resolution data
                        with rasterio.open(
                            output_path,
                            'w',
                            driver='GTiff',
                            height=merged_data.shape[0],
                            width=merged_data.shape[1],
                            count=1,
                            dtype=merged_data.dtype,
                            crs='+proj=latlong',
                            transform=transform,
                        ) as dst:
                            dst.write(merged_data, 1)
                    
                    logger.info(f"Merged SRTM data saved to {output_path}")
                    return output_path
            except Exception as e:
                logger.error(f"Error creating merged SRTM data: {e}")
                return None
    except Exception as e:
        logger.error(f"Error downloading NASA SRTM data: {e}")
        return None

@cache_result(expire_seconds=86400*7)  # Cache for 7 days
def download_sample_lidar(
    dataset_id: str,
    bounds: Tuple[float, float, float, float],
    output_path: Optional[Path] = None,
    resolution: int = SAMPLE_RESOLUTION,
    region_name: Optional[str] = None
) -> Optional[Path]:
    """
    Download LiDAR data for the specified region from various sources.
    
    Sources include:
    - OpenTopography API (standard global datasets)
    - NASA SRTM data (global elevation)
    - Specialized Amazon LiDAR datasets (research-specific)
    
    Falls back to SRTM data if the specific dataset_id is not available.
    
    Args:
        dataset_id: Dataset ID (OpenTopography ID, 'srtm', or Amazon dataset ID)
        bounds: (min_lon, min_lat, max_lon, max_lat)
        output_path: Path to save the downloaded file (None for auto-generation)
        resolution: Output resolution in meters
        region_name: Optional specific region name within a dataset
        
    Returns:
        Path to downloaded file or None if failed
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # Generate output path if not provided
    if output_path is None:
        # Create a safe filename based on bounds and dataset
        filename = f"lidar_{dataset_id}_{min_lon}_{min_lat}_{max_lon}_{max_lat}_{resolution}m.tif"
        output_path = LIDAR_DATA_PATH / filename
    
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if file already exists
    if output_path.exists():
        if os.path.getsize(output_path) > 10000:  # Make sure it's not a tiny error file
            logger.info(f"LiDAR file already exists at {output_path}")
            return output_path
        else:
            # Remove potentially corrupted file
            logger.warning(f"Removing small/potentially corrupted file: {output_path}")
            os.remove(output_path)
    
    # Check if we should use SRTM data (global coverage)
    if dataset_id.lower() == 'srtm':
        logger.info(f"Using direct SRTM download for region {bounds}")
        srtm_result = download_nasa_srtm(bounds, output_path, resolution)
        if srtm_result:
            logger.info(f"SRTM data successfully downloaded to {srtm_result}")
            return srtm_result
        else:
            logger.error(f"Failed to download SRTM data for region {bounds}")
            return None
    
    # Check if this is an Amazon-specific dataset
    if dataset_id.startswith(('ornl_', 'zenodo_', 'embrapa_')):
        try:
            from .amazon_lidar import download_amazon_lidar_sample, get_amazon_lidar_metadata
            
            # Check if the dataset exists
            metadata = get_amazon_lidar_metadata(dataset_id)
            if metadata:
                logger.info(f"Using specialized Amazon LiDAR dataset: {dataset_id}")
                amazon_result = download_amazon_lidar_sample(
                    dataset_id, 
                    bounds, 
                    output_path, 
                    resolution,
                    region_name
                )
                if amazon_result:
                    logger.info(f"Amazon LiDAR data information available at {amazon_result}")
                    return amazon_result
                else:
                    logger.warning(f"Failed to access Amazon LiDAR data, falling back to other sources")
        except ImportError:
            logger.warning("Amazon LiDAR module not available")
    
    # Try OpenTopography API for specific dataset
    api_url = "https://portal.opentopography.org/API/otCatalog"
    
    params = {
        "datasetId": dataset_id,
        "south": min_lat,
        "north": max_lat,
        "west": min_lon,
        "east": max_lon, 
        "outputFormat": "GTiff",
        "API_Key": OPEN_TOPO_KEY
    }
    
    logger.info(f"Downloading data from OpenTopography dataset {dataset_id} for region {bounds} using API key")
    
    try:
        response = requests.get(api_url, params=params, stream=True, timeout=60)
        response.raise_for_status()
        
        # Save the file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify file was properly created
        if os.path.getsize(output_path) < 1000:  # Less than 1KB suggests error
            with open(output_path, 'r') as f:
                content = f.read()
                if "error" in content.lower():
                    logger.error(f"API returned error: {content}")
                    # Try SRTM instead
                    logger.info("Falling back to SRTM data")
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    return download_nasa_srtm(bounds, output_path, resolution)
        
        logger.info(f"LiDAR data downloaded to {output_path}")
        return output_path
    
    except requests.RequestException as e:
        logger.error(f"Error downloading data from OpenTopography: {e}")
        # Try SRTM instead
        logger.info("Falling back to SRTM data after OpenTopography error")
        return download_nasa_srtm(bounds, output_path, resolution)

def _split_region_into_tiles(
    bounds: Tuple[float, float, float, float], 
    max_size_degrees: float = MAX_REGION_SIZE_DEGREES
) -> List[Tuple[float, float, float, float]]:
    """
    Split a large region into smaller tiles to avoid size limits in Earth Engine.
    
    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat)
        max_size_degrees: Maximum size of a tile in degrees
        
    Returns:
        List of tile bounds (min_lon, min_lat, max_lon, max_lat)
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    width = max_lon - min_lon
    height = max_lat - min_lat
    
    # If region is small enough, return as is
    if width <= max_size_degrees and height <= max_size_degrees:
        return [bounds]
    
    # Calculate number of tiles needed in each direction
    num_x_tiles = max(1, math.ceil(width / max_size_degrees))
    num_y_tiles = max(1, math.ceil(height / max_size_degrees))
    
    # Calculate tile sizes
    x_tile_size = width / num_x_tiles
    y_tile_size = height / num_y_tiles
    
    # Generate tiles
    tiles = []
    for i in range(num_x_tiles):
        for j in range(num_y_tiles):
            tile_min_lon = min_lon + i * x_tile_size
            tile_min_lat = min_lat + j * y_tile_size
            tile_max_lon = min_lon + (i + 1) * x_tile_size
            tile_max_lat = min_lat + (j + 1) * y_tile_size
            
            # Ensure we don't exceed original bounds due to floating point errors
            tile_max_lon = min(tile_max_lon, max_lon)
            tile_max_lat = min(tile_max_lat, max_lat)
            
            tiles.append((tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat))
    
    logger.info(f"Split region {bounds} into {len(tiles)} tiles for processing")
    return tiles

@cache_result(expire_seconds=86400*7)  # Cache for 7 days
def fetch_sentinel_imagery(
    bounds: Tuple[float, float, float, float],
    start_date: str = None,
    end_date: str = None,
    cloud_cover_max: int = 20,
    bands: List[str] = ["B2", "B3", "B4", "B8"],
    output_path: Optional[Path] = None
) -> Optional[Path]:
    """
    Fetch Sentinel-2 imagery using Google Earth Engine (free).
    Downloads a composite image with specified bands at medium resolution.
    For large regions, uses adaptive resolution to avoid size limits.
    
    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat)
        start_date: Start date in format 'YYYY-MM-DD' (default: 6 months ago)
        end_date: End date in format 'YYYY-MM-DD' (default: today)
        cloud_cover_max: Maximum cloud cover percentage
        bands: List of Sentinel-2 bands to include
        output_path: Path to save the downloaded file (None for auto-generation)
        
    Returns:
        Path to downloaded file or None if failed
    """
    if not GEE_AVAILABLE:
        logger.error("Google Earth Engine is not initialized. Cannot fetch Sentinel imagery.")
        return None
    
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        start_date_dt = datetime.now() - timedelta(days=180)  # 6 months ago
        start_date = start_date_dt.strftime('%Y-%m-%d')
    
    # Generate output path if not provided
    if output_path is None:
        # Create a safe filename based on bounds and dates
        min_lon, min_lat, max_lon, max_lat = bounds
        filename = f"sentinel_{min_lon}_{min_lat}_{max_lon}_{max_lat}_{start_date}_{end_date}.tif"
        output_path = SATELLITE_DATA_PATH / filename
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if file already exists
    if output_path.exists():
        if os.path.getsize(output_path) > 10000:  # Make sure it's not a tiny error file
            logger.info(f"Sentinel imagery already exists at {output_path}")
            return output_path
        else:
            # Remove potentially corrupted file
            logger.warning(f"Removing small/potentially corrupted file: {output_path}")
            os.remove(output_path)
    
    # Try to download imagery with multiple attempts and fallback options
    try:
        # First attempt - try normal download
        result = _download_sentinel_tile(bounds, start_date, end_date, cloud_cover_max, bands, output_path)
        if result:
            return result
            
        # If that fails, try with increased cloud cover
        logger.warning("First download attempt failed. Trying with increased cloud cover.")
        increased_cloud = min(cloud_cover_max * 2, 90)
        result = _download_sentinel_tile(bounds, start_date, end_date, increased_cloud, bands, output_path)
        if result:
            return result
            
        # If still fails, try with longer time range
        logger.warning("Download with increased cloud cover failed. Trying with longer time range.")
        extended_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
        result = _download_sentinel_tile(bounds, extended_start, end_date, increased_cloud, bands, output_path)
        if result:
            return result
            
        # Last resort - try with minimal bands (RGB only) and lower resolution
        logger.warning("All previous attempts failed. Trying minimal bands and lower resolution.")
        minimal_bands = ["B4", "B3", "B2"]  # RGB only
        result = _download_sentinel_tile(bounds, extended_start, end_date, 90, minimal_bands, output_path, force_low_res=True)
        
        return result
    except Exception as e:
        logger.error(f"All download attempts failed: {e}")
        return None

def _download_sentinel_tile(
    bounds: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    cloud_cover_max: int,
    bands: List[str],
    output_path: Path,
    force_low_res: bool = False
) -> Optional[Path]:
    """
    Download a single Sentinel-2 tile.
    
    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat)
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'
        cloud_cover_max: Maximum cloud cover percentage
        bands: List of Sentinel-2 bands to include
        output_path: Path to save the downloaded file
        force_low_res: Whether to force low resolution
        
    Returns:
        Path to downloaded file or None if failed
    """
    try:
        # Create region of interest
        roi = ee.Geometry.Rectangle(bounds)
        
        # Load Sentinel-2 collection
        collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                     .filterDate(start_date, end_date)
                     .filterBounds(roi)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_max)))
        
        # Check if we have any images
        collection_size = collection.size().getInfo()
        if collection_size == 0:
            logger.warning(f"No Sentinel images found for region {bounds} with cloud cover <= {cloud_cover_max}%")
            return None
        else:
            logger.info(f"Found {collection_size} Sentinel images for region {bounds}")
        
        # Create RGB composite
        composite = collection.median().select(bands)
        
        # Get a temporary file for the download
        temp_fd, temp_file = tempfile.mkstemp(suffix='.tif')
        os.close(temp_fd)
        temp_path = Path(temp_file)
        
        # Calculate scale based on region size to avoid too large requests
        # Larger regions get lower resolution
        min_lon, min_lat, max_lon, max_lat = bounds
        width_degrees = max_lon - min_lon
        height_degrees = max_lat - min_lat
        
        # Calculate approximate region size in km
        # 1 degree of longitude at the equator is ~111 km, but varies with latitude
        # cos(latitude in radians) gives the factor to multiply longitude distance
        import math
        avg_lat_radians = math.radians((min_lat + max_lat) / 2)
        lon_distance_factor = math.cos(avg_lat_radians)
        
        # Calculate distances in km
        width_km = width_degrees * 111 * lon_distance_factor
        height_km = height_degrees * 111  # Latitude degrees are consistent
        
        region_size_km = max(width_km, height_km)
        
        # Adaptive scaling with minimum resolution for archaeological features
        # For small regions (<10km), use 20m resolution (good for archaeological features)
        # For medium regions (10-50km), scale from 20m to 60m
        # For large regions (>50km), use 60m+ resolution
        
        if force_low_res:
            # Use lower resolution when requested
            adjusted_scale = 100  # 100m resolution for difficult downloads
        elif region_size_km <= 10:
            adjusted_scale = 20  # High resolution for small areas
        elif region_size_km <= 50:
            # Linear scaling from 20m to 60m
            adjusted_scale = 20 + (region_size_km - 10) * (40 / 40)
        else:
            # For very large regions, scale more aggressively but cap at 100m
            adjusted_scale = min(60 + (region_size_km - 50) * 0.8, 100)
        
        adjusted_scale = int(adjusted_scale)
        logger.info(f"Using resolution of {adjusted_scale}m for region size {region_size_km:.1f}km")
        
        # Get the download URL with adjusted scale
        try:
            url = composite.getDownloadURL({
                'scale': adjusted_scale,
                'crs': 'EPSG:4326',
                'region': roi,
                'format': 'GEO_TIFF'
            })
        except ee.ee_exception.EEException as e:
            if "Total request size" in str(e):
                # Double the scale and retry
                adjusted_scale = adjusted_scale * 2
                logger.warning(f"Request too large. Retrying with lower resolution: {adjusted_scale}m")
                
                url = composite.getDownloadURL({
                    'scale': adjusted_scale,
                    'crs': 'EPSG:4326',
                    'region': roi,
                    'format': 'GEO_TIFF'
                })
            else:
                raise
        
        # Download the file with retry logic
        max_retries = 3
        retry_count = 0
        download_success = False
        
        while retry_count < max_retries and not download_success:
            try:
                # Download the file
                logger.info(f"Downloading Sentinel imagery from URL: {url}")
                response = requests.get(url, stream=True, timeout=300)  # 5-minute timeout
                response.raise_for_status()
                
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Check if the download is valid
                if os.path.getsize(temp_path) < 10000:  # Less than 10KB suggests error
                    logger.warning(f"Downloaded file is too small ({os.path.getsize(temp_path)} bytes). Retrying...")
                    retry_count += 1
                    continue
                
                # Move temp file to final destination
                shutil.move(temp_path, output_path)
                download_success = True
                logger.info(f"Sentinel imagery saved to {output_path}")
                
            except requests.RequestException as e:
                logger.error(f"Download error (attempt {retry_count+1}/{max_retries}): {e}")
                retry_count += 1
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                time.sleep(5)  # Wait before retrying
        
        if download_success:
            return output_path
        else:
            logger.error(f"Failed to download after {max_retries} attempts")
            return None
    
    except ee.ee_exception.EEException as e:
        if "Total request size" in str(e) and not force_low_res:
            # If the request is too large, try with a much lower resolution
            logger.warning(f"Request too large. Retrying with much lower resolution (force_low_res=True)")
            return _download_sentinel_tile(
                bounds, 
                start_date, 
                end_date, 
                cloud_cover_max, 
                bands, 
                output_path, 
                force_low_res=True
            )
        else:
            logger.error(f"Earth Engine error: {e}")
            return None
    except Exception as e:
        logger.error(f"Error downloading Sentinel tile: {e}")
        return None

def get_data_sources_for_region(
    bounds: Tuple[float, float, float, float]
) -> Dict:
    """
    Get available free data sources for a region.
    Combines metadata from multiple sources to inform data acquisition.
    
    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat)
        
    Returns:
        Dictionary of available data sources
    """
    # Get LiDAR metadata
    lidar_metadata = fetch_lidar_metadata(bounds)
    
    # Check if Sentinel data is available for the region
    sentinel_available = GEE_AVAILABLE
    
    # Check if NASA SRTM data is available (covers most of Earth)
    srtm_available = (
        bounds[1] >= -56 and bounds[3] <= 60  # SRTM covers latitude -56 to 60
    )
    
    # Check for specialized Amazon LiDAR datasets
    try:
        from .amazon_lidar import get_amazon_lidar_sources_for_region
        amazon_lidar_sources = get_amazon_lidar_sources_for_region(bounds)
        amazon_lidar_available = len(amazon_lidar_sources) > 0
    except ImportError:
        logger.warning("Amazon LiDAR module not available")
        amazon_lidar_sources = []
        amazon_lidar_available = False
    
    # Return combined metadata
    return {
        "lidar": {
            "available": len(lidar_metadata) > 0,
            "sources": lidar_metadata,
            "count": len(lidar_metadata)
        },
        "sentinel": {
            "available": sentinel_available
        },
        "srtm": {
            "available": srtm_available
        },
        "amazon_lidar": {
            "available": amazon_lidar_available,
            "sources": amazon_lidar_sources,
            "count": len(amazon_lidar_sources)
        }
    }

def fetch_high_res_imagery_for_coordinate(
    lat: float,
    lon: float,
    radius_meters: int = 500,
    start_date: str = None,
    end_date: str = None,
    output_path: Optional[Path] = None
) -> Optional[Path]:
    """
    Fetch high-resolution satellite imagery for a specific coordinate.
    Uses Sentinel-2 imagery at 10m resolution.
    
    Args:
        lat: Latitude of the point of interest
        lon: Longitude of the point of interest
        radius_meters: Radius around the point to fetch (in meters)
        start_date: Start date in format 'YYYY-MM-DD' (default: 2 years ago)
        end_date: End date in format 'YYYY-MM-DD' (default: today)
        output_path: Path to save the downloaded file (None for auto-generation)
        
    Returns:
        Path to downloaded file or None if failed
    """
    if not GEE_AVAILABLE:
        logger.error("Google Earth Engine is not initialized. Cannot fetch imagery.")
        return None
    
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        start_date_dt = datetime.now() - timedelta(days=730)  # 2 years ago for better coverage
        start_date = start_date_dt.strftime('%Y-%m-%d')
    
    # Generate output path if not provided
    if output_path is None:
        # Create a safe filename based on coords and dates
        filename = f"highres_{lat}_{lon}_{radius_meters}m_{start_date}_{end_date}.tif"
        output_path = SATELLITE_DATA_PATH / filename
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if file already exists
    if output_path.exists():
        if os.path.getsize(output_path) > 10000:  # Make sure it's not a tiny error file
            logger.info(f"High-resolution imagery already exists at {output_path}")
            return output_path
        else:
            # Remove potentially corrupted file
            logger.warning(f"Removing small/potentially corrupted file: {output_path}")
            os.remove(output_path)
    
    try:
        # Create point and buffer
        point = ee.Geometry.Point([lon, lat])
        buffer = point.buffer(radius_meters)
        
        # Load Sentinel-2 collection
        sentinel = ee.ImageCollection('COPERNICUS/S2') \
            .filterDate(start_date, end_date) \
            .filterBounds(buffer) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        
        # Check if we found any images
        count = sentinel.size().getInfo()
        if count == 0:
            logger.warning(f"No Sentinel-2 images found for point ({lat}, {lon}) in date range {start_date} to {end_date}")
            return None
            
        logger.info(f"Found {count} Sentinel-2 images")
        
        # Get the most recent, least cloudy image (instead of median)
        image_list = sentinel.sort('CLOUDY_PIXEL_PERCENTAGE').toList(1)
        single_image = ee.Image(image_list.get(0))
        
        # Apply visualization parameters directly in getThumbURL instead of using visualize
        thumb_url = single_image.getThumbURL({
            'dimensions': 1024,
            'region': buffer,
            'format': 'png',
            'bands': ['B4', 'B3', 'B2'],
            'min': 0,
            'max': 3000
        })
        
        # Download the thumbnail
        logger.info(f"Downloading high-resolution thumbnail from Sentinel-2")
        response = requests.get(thumb_url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Save as PNG temporarily
        temp_png = output_path.with_suffix('.png')
        with open(temp_png, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        # Check if download succeeded
        if not os.path.exists(temp_png) or os.path.getsize(temp_png) < 10000:
            logger.error(f"Failed to download high-resolution thumbnail (file too small or missing)")
            if os.path.exists(temp_png):
                os.remove(temp_png)
            return None
            
        # Convert PNG to TIFF with geo-referencing
        # We'll create a simple TIFF with the PNG content since we don't need precise georeferencing
        # for verification purposes
        
        # Approximate bounds of the buffer
        xmin = lon - (radius_meters / 111320)
        xmax = lon + (radius_meters / 111320)
        ymin = lat - (radius_meters / (111320 * math.cos(math.radians(lat))))
        ymax = lat + (radius_meters / (111320 * math.cos(math.radians(lat))))
        
        # Read PNG with rasterio and create a new GeoTIFF
        try:
            import PIL.Image
            import numpy as np
            
            # Open the PNG file
            img = PIL.Image.open(temp_png)
            img_array = np.array(img)
            
            # Create new GeoTIFF
            height, width, bands = img_array.shape
            transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)
            
            # Write to GeoTIFF
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=bands,
                dtype=img_array.dtype,
                crs='+proj=longlat +datum=WGS84 +no_defs',
                transform=transform
            ) as dst:
                for i in range(bands):
                    dst.write(img_array[:, :, i], i+1)
            
            # Clean up temporary PNG
            os.remove(temp_png)
            
            logger.info(f"Successfully created high-resolution imagery at {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error converting PNG to GeoTIFF: {e}")
            # Just return the PNG if conversion fails
            logger.info(f"Using PNG instead of GeoTIFF due to conversion error")
            os.rename(temp_png, output_path.with_suffix('.png'))
            return output_path.with_suffix('.png')
        
    except Exception as e:
        logger.error(f"Error fetching high-resolution imagery: {e}")
        return None 