"""
Feature detection module for identifying potential archaeological sites.
Uses cost-effective algorithms to preprocess data before applying AI models.
"""

import os
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
from datetime import datetime

import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, Point, Polygon, MultiPolygon
import cv2
from skimage import filters, feature, morphology, segmentation

from ..config import (
    INITIAL_REGION_BOUNDS,
    SAMPLE_RESOLUTION,
    DATA_DIR,
    LIDAR_DATA_PATH
)
from ..utils.cache import cache_result

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@cache_result(expire_seconds=86400*7)  # Cache for 7 days
def detect_geometric_features(
    lidar_path: Path,
    min_size: int = 80,  # Minimum feature size in meters
    max_size: int = 500,  # Maximum feature size in meters
    shapes_to_detect: List[str] = ["rectangle", "circle", "line"],
    sensitivity: float = 0.5  # Detection sensitivity (0-1)
) -> List[Dict]:
    """
    Detect geometric shapes in LiDAR data that might indicate archaeological features.
    Uses classic computer vision approaches to minimize computational costs.
    
    Args:
        lidar_path: Path to LiDAR GeoTIFF
        min_size: Minimum feature size in meters
        max_size: Maximum feature size in meters
        shapes_to_detect: Which shape types to detect
        sensitivity: Detection sensitivity (0-1)
        
    Returns:
        List of detected features with metadata
    """
    try:
        lidar_path = Path(lidar_path)
        
        # Check if we have a guidance file instead of actual data
        if str(lidar_path).endswith('.guidance.txt'):
            logger.info(f"Guidance file provided instead of LiDAR data: {lidar_path}")
            logger.info(f"Skipping feature detection for guidance files.")
            
            # Print the guidance file contents
            try:
                with open(lidar_path, 'r') as f:
                    logger.info(f"Guidance file contents (first 10 lines):")
                    for i, line in enumerate(f):
                        if i < 10:
                            logger.info(f"  {line.strip()}")
                        else:
                            break
                            
                # Try to fall back to SRTM data for this region
                from ..preprocessing.data_fetcher import download_nasa_srtm
                
                # Extract bounds from guidance filename
                # Filename format: amazon_lidar_dataset_id_min_lon_min_lat_max_lon_max_lat_resolution.guidance.txt
                parts = lidar_path.name.split('_')
                if len(parts) >= 8:
                    try:
                        # Parse bounds from filename
                        min_lon = float(parts[-6])
                        min_lat = float(parts[-5])
                        max_lon = float(parts[-4])
                        max_lat = float(parts[-3])
                        resolution = parts[-2].replace('m', '')
                        
                        bounds = (min_lon, min_lat, max_lon, max_lat)
                        logger.info(f"Extracted bounds from guidance filename: {bounds}")
                        
                        # Try to get SRTM data as a fallback
                        srtm_filename = f"lidar_srtm_{min_lon}_{min_lat}_{max_lon}_{max_lat}_{resolution}m.tif"
                        srtm_path = LIDAR_DATA_PATH / srtm_filename
                        
                        if os.path.exists(srtm_path):
                            logger.info(f"Using existing SRTM data for feature detection: {srtm_path}")
                            return detect_geometric_features(srtm_path, min_size, max_size, shapes_to_detect, sensitivity)
                        else:
                            logger.info(f"No SRTM fallback available. Returning empty feature list.")
                    except (ValueError, IndexError) as e:
                        logger.error(f"Error parsing bounds from guidance filename: {e}")
            except Exception as e:
                logger.error(f"Error reading guidance file: {e}")
                
            return []
        
        # Open LiDAR raster
        with rasterio.open(lidar_path) as src:
            # Read elevation data
            elevation = src.read(1)
            
            # Get transformation info
            transform = src.transform
            
            # Convert transform units to meters
            # For GeoTIFFs in geographic coordinates (EPSG:4326), we need to approximate meters
            is_geographic = src.crs and (src.crs.to_epsg() == 4326 or 'EPSG:4326' in str(src.crs))
            
            if is_geographic:
                # Get approximate center latitude for scaling
                bounds = src.bounds
                center_lat = (bounds.bottom + bounds.top) / 2
                
                # Convert degrees to approximate meters
                import math
                # At the equator, 1 degree longitude ≈ 111,320 meters
                # At other latitudes, multiply by cos(latitude)
                lon_scale = 111320 * math.cos(math.radians(center_lat))
                lat_scale = 111320  # 1 degree latitude is always ~111,320 meters
                
                # Pixel dimensions in meters (approximate)
                pixel_width_m = abs(transform[0] * lon_scale)
                pixel_height_m = abs(transform[4] * lat_scale)
                
                logger.info(f"Geographic coordinates detected. Approximating pixel size: {pixel_width_m:.1f}m x {pixel_height_m:.1f}m")
            else:
                # For projected coordinates, units are already in meters
                pixel_width_m = abs(transform[0])
                pixel_height_m = abs(transform[4])
            
            # Calculate minimum and maximum feature sizes in pixels
            min_size_px = max(3, int(min_size / max(pixel_width_m, pixel_height_m)))
            max_size_px = int(max_size / max(pixel_width_m, pixel_height_m))
            
            # Adjust for sensitivity - lower thresholds with higher sensitivity
            edge_low_threshold = max(0.05, 0.2 - (sensitivity * 0.15))
            edge_high_threshold = max(0.1, 0.3 - (sensitivity * 0.15))
            
            # Check if the resolution is too coarse for meaningful detection
            if pixel_width_m > 100 or pixel_height_m > 100:
                logger.warning(f"Resolution too coarse for geometric detection: {pixel_width_m}m x {pixel_height_m}m")
                # Even with coarse data, try with relaxed parameters for archaeological features
                min_size_px = max(2, min_size_px // 2)
            
            # Normalize elevation data for better edge detection
            if np.ptp(elevation) > 0:  # Ensure range is not zero
                elevation_norm = (elevation - np.min(elevation)) / np.ptp(elevation)
            else:
                elevation_norm = np.zeros_like(elevation)
            
            # Preprocessing: fill small holes and smooth
            try:
                elevation_filled = morphology.closing(elevation_norm, morphology.disk(2))
                elevation_smooth = filters.gaussian(elevation_filled, sigma=1.5)
                
                # Calculate local relief model (highlight local variations)
                # For Amazon geoglyphs, which can be extremely subtle, use different sigma
                # values based on sensitivity to better highlight faint earthworks
                local_relief_sigma = 10
                if sensitivity > 0.7:
                    # For high sensitivity, use a smaller sigma to detect finer details
                    local_relief_sigma = 5
                
                elevation_trend = filters.gaussian(elevation_smooth, sigma=local_relief_sigma)
                local_relief = elevation_smooth - elevation_trend
                
                # Calculate slope-based enhancement (geoglyphs often visible in slope analysis)
                elevation_sobelx = filters.sobel_h(elevation_smooth)
                elevation_sobely = filters.sobel_v(elevation_smooth)
                slope = np.sqrt(elevation_sobelx**2 + elevation_sobely**2)
                
                # Enhance contrast with percentile-based normalization
                # Use more extreme percentiles for higher sensitivity
                p_low = max(1, 10 - int(sensitivity * 9))  # 1-10% based on sensitivity
                p_high = min(99, 90 + int(sensitivity * 9))  # 90-99% based on sensitivity
                
                p_low_val, p_high_val = np.percentile(local_relief, (p_low, p_high))
                if p_high_val > p_low_val:
                    local_relief_enhanced = np.clip((local_relief - p_low_val) / (p_high_val - p_low_val), 0, 1)
                else:
                    local_relief_enhanced = local_relief
                
                # Combine local relief and slope for better detection of subtle features
                if sensitivity > 0.7:
                    # Normalize and weight slope
                    slope_norm = (slope - np.min(slope)) / (np.max(slope) - np.min(slope)) if np.ptp(slope) > 0 else slope
                    # Weight based on sensitivity
                    weight = 0.5 + (sensitivity - 0.5) * 0.5  # 0.5-0.75 for sensitivity 0.5-1.0
                    # Combine (weight * local_relief + (1-weight) * slope)
                    combined_relief = weight * local_relief_enhanced + (1 - weight) * slope_norm
                else:
                    combined_relief = local_relief_enhanced
                
                # Detect edges using Canny edge detection with sensitivity-adjusted thresholds
                edges = feature.canny(
                    combined_relief, 
                    sigma=1.0,
                    low_threshold=edge_low_threshold,
                    high_threshold=edge_high_threshold
                )
                
                # Additional morphological operations to connect broken edges
                # More important for archaeological features which may be degraded
                if sensitivity > 0.6:
                    edges = morphology.binary_dilation(edges, morphology.disk(1))
                    edges = morphology.binary_erosion(edges, morphology.disk(1))
                
            except Exception as e:
                logger.error(f"Error in preprocessing: {e}")
                return []
            
            # Convert to uint8 for OpenCV operations
            edges_uint8 = (edges * 255).astype(np.uint8)
            
            # Find contours
            try:
                contours, _ = cv2.findContours(
                    edges_uint8, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
            except Exception as e:
                logger.error(f"Error finding contours: {e}")
                return []
            
            # Initialize results list
            features = []
            
            # Check if any contours were found
            if not contours or len(contours) == 0:
                logger.warning(f"No contours found in elevation data")
                return []
            
            # Process each contour
            for contour in contours:
                # Skip if contour is too small
                if len(contour) < 5:
                    continue
                
                # Filter by size with sensitivity-adjusted thresholds
                area = cv2.contourArea(contour)
                min_area_threshold = (min_size_px * min_size_px * 0.25) * (1.0 - (sensitivity * 0.5))
                max_area_threshold = (max_size_px * max_size_px) * (1.0 + (sensitivity * 0.5))
                
                if area < min_area_threshold or area > max_area_threshold:
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if too small - more lenient with higher sensitivity
                min_dimension_px = min_size_px * (0.5 - (sensitivity * 0.2))
                if w < min_dimension_px or h < min_dimension_px:
                    continue
                
                # Determine shape type
                shape_type = "unknown"
                shape_confidence = 0.0
                
                # Rectangle detection
                if "rectangle" in shapes_to_detect:
                    # Approximate polygon
                    # Use sensitivity to adjust epsilon - lower epsilon with higher sensitivity
                    # to detect more subtle rectangular features
                    epsilon = 0.02 * cv2.arcLength(contour, True) * (1.0 - (sensitivity * 0.5))
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Check if it's a rectangle (4 points, or 5-6 for imperfect ones with high sensitivity)
                    if len(approx) >= 4 and len(approx) <= (4 + int(sensitivity * 2)):
                        shape_type = "rectangle"
                        # Higher confidence for exact rectangles, lower for approximations
                        shape_confidence = 0.8 - (0.1 * (len(approx) - 4))
                        
                        # Check if it's a square
                        aspect_ratio = float(w) / h if h > 0 else 0
                        if 0.8 <= aspect_ratio <= 1.2:
                            shape_type = "square"
                            shape_confidence = 0.9
                
                # Circle detection
                if "circle" in shapes_to_detect and shape_type == "unknown":
                    # Fit a circle
                    (x_c, y_c), radius = cv2.minEnclosingCircle(contour)
                    circle_area = np.pi * (radius ** 2)
                    
                    # Check how well it matches a circle, with sensitivity-adjusted threshold
                    min_circularity = 0.7 - (sensitivity * 0.2)  # 0.5 to 0.7 based on sensitivity
                    circularity = area / circle_area if circle_area > 0 else 0
                    if circularity > min_circularity:
                        shape_type = "circle"
                        shape_confidence = circularity
                
                # Line detection (linear features like walls)
                if "line" in shapes_to_detect and shape_type == "unknown":
                    # Check if elongated
                    aspect_ratio = float(max(w, h)) / min(w, h) if min(w, h) > 0 else 0
                    min_line_ratio = 3.0 - (sensitivity * 1.0)  # 2.0 to 3.0 based on sensitivity
                    if aspect_ratio > min_line_ratio:
                        shape_type = "line"
                        shape_confidence = min(aspect_ratio / 10, 0.9)  # Cap at 0.9
                
                # If no specific shape detected, use generic feature
                if shape_type == "unknown":
                    shape_type = "feature"
                    # Calculate compactness as a confidence measure
                    perimeter = cv2.arcLength(contour, True)
                    compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    shape_confidence = min(compactness, 0.7)  # Cap at 0.7 for unknown shapes
                
                # Safety check - ensure we're within bounds for real-world data
                if y < 0 or y >= elevation.shape[0] or x < 0 or x >= elevation.shape[1]:
                    continue
                
                # Convert pixel coordinates to geographic coordinates
                try:
                    lon, lat = rasterio.transform.xy(transform, y + h/2, x + w/2)
                except Exception as e:
                    logger.error(f"Error converting coordinates: {e}")
                    continue
                
                # Calculate size in meters
                width_m = w * pixel_width_m
                height_m = h * pixel_height_m
                
                # Get elevation statistics for the feature
                try:
                    feature_mask = np.zeros_like(elevation, dtype=np.uint8)
                    cv2.drawContours(feature_mask, [contour], 0, 1, -1)
                    elevation_values = elevation[feature_mask > 0]
                    
                    if len(elevation_values) > 0:
                        elev_min = float(np.min(elevation_values))
                        elev_max = float(np.max(elevation_values))
                        elev_mean = float(np.mean(elevation_values))
                        elev_range = float(elev_max - elev_min)
                    else:
                        elev_min = elev_max = elev_mean = 0
                        elev_range = 0
                except Exception as e:
                    logger.error(f"Error calculating elevation statistics: {e}")
                    elev_min = elev_max = elev_mean = elev_range = 0
                
                # Create feature dictionary
                feature_dict = {
                    "type": shape_type,
                    "confidence": float(shape_confidence),
                    "coordinates": {
                        "lon": float(lon),
                        "lat": float(lat)
                    },
                    "size": {
                        "width_m": float(width_m),
                        "height_m": float(height_m),
                        "area_m2": float(width_m * height_m)
                    },
                    "elevation": {
                        "min": elev_min,
                        "max": elev_max,
                        "mean": elev_mean,
                        "range": elev_range
                    },
                    "shape": shape_type,
                    "metadata": {
                        "source": "lidar",
                        "detection_method": "geometric",
                        "file": str(lidar_path.name)
                    }
                }
                
                features.append(feature_dict)
            
            logger.info(f"Detected {len(features)} potential geometric features")
            return features
    
    except Exception as e:
        logger.error(f"Error detecting geometric features: {e}")
        return []

@cache_result(expire_seconds=86400*7)  # Cache for 7 days
def detect_vegetation_anomalies(
    satellite_path: Path,
    min_size: int = 80,  # Minimum feature size in meters
    max_size: int = 500,  # Maximum feature size in meters,
    sensitivity: float = 0.5  # Detection sensitivity (0-1)
) -> List[Dict]:
    """
    Detect vegetation anomalies that might indicate archaeological features.
    Looks for patterns in vegetation that suggest anthropogenic influence.
    
    Args:
        satellite_path: Path to satellite image GeoTIFF
        min_size: Minimum feature size in meters
        max_size: Maximum feature size in meters
        sensitivity: Detection sensitivity (0-1)
        
    Returns:
        List of detected features with metadata
    """
    try:
        # Open satellite image
        with rasterio.open(satellite_path) as src:
            # Read all bands
            bands = src.read()
            
            # Get transformation info
            transform = src.transform
            
            # Convert transform units to meters
            # For GeoTIFFs in geographic coordinates (EPSG:4326), we need to approximate meters
            is_geographic = src.crs and (src.crs.to_epsg() == 4326 or 'EPSG:4326' in str(src.crs))
            
            if is_geographic:
                # Get approximate center latitude for scaling
                bounds = src.bounds
                center_lat = (bounds.bottom + bounds.top) / 2
                
                # Convert degrees to approximate meters
                import math
                # At the equator, 1 degree longitude ≈ 111,320 meters
                # At other latitudes, multiply by cos(latitude)
                lon_scale = 111320 * math.cos(math.radians(center_lat))
                lat_scale = 111320  # 1 degree latitude is always ~111,320 meters
                
                # Pixel dimensions in meters (approximate)
                pixel_width_m = abs(transform[0] * lon_scale)
                pixel_height_m = abs(transform[4] * lat_scale)
                
                logger.info(f"Geographic coordinates detected. Approximating pixel size: {pixel_width_m:.1f}m x {pixel_height_m:.1f}m")
            else:
                # For projected coordinates, units are already in meters
                pixel_width_m = abs(transform[0])
                pixel_height_m = abs(transform[4])
            
            # Print debug info about the satellite data
            logger.info(f"Satellite image has {bands.shape[0]} bands with shape {bands.shape[1]}x{bands.shape[2]} at resolution {pixel_width_m:.1f}m x {pixel_height_m:.1f}m")
            
            # Skip if resolution is too coarse
            if pixel_width_m > 100 or pixel_height_m > 100:
                logger.warning(f"Resolution too coarse for vegetation analysis: {pixel_width_m:.1f}m x {pixel_height_m:.1f}m")
                return []
            
            # Calculate minimum and maximum feature sizes in pixels
            min_size_px = max(3, int(min_size / max(pixel_width_m, pixel_height_m)))
            max_size_px = int(max_size / max(pixel_width_m, pixel_height_m))
            
            # Boost sensitivity for archaeological features 
            # (which can be subtle in the Amazon)
            adjusted_sensitivity = min(sensitivity * 1.5, 0.95)
            
            # Enhanced vegetation indices for archaeological feature detection
            # We'll calculate multiple indices to improve detection rate
            indices = {}
            
            # Check if we have enough bands for indices
            if bands.shape[0] >= 4:
                # Ensure bands have valid data
                for i in range(bands.shape[0]):
                    if np.all(bands[i] == 0) or np.all(np.isnan(bands[i])):
                        logger.warning(f"Band {i+1} has no valid data, using placeholder")
                        bands[i] = np.ones_like(bands[i]) * 0.1  # Small positive value as placeholder
                
                # Assuming order: B, G, R, NIR for Sentinel-2 (adjust if different)
                blue = bands[0].astype(float)
                green = bands[1].astype(float)
                red = bands[2].astype(float)
                nir = bands[3].astype(float)
                
                # Calculate NDVI (Normalized Difference Vegetation Index)
                ndvi = np.zeros_like(red)
                valid_mask = (red + nir) > 0
                ndvi[valid_mask] = (nir[valid_mask] - red[valid_mask]) / (nir[valid_mask] + red[valid_mask])
                indices["ndvi"] = ndvi
                
                # Calculate EVI (Enhanced Vegetation Index) - better in dense vegetation
                # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
                evi_denom = nir + 6.0 * red - 7.5 * blue + 1.0
                evi = np.zeros_like(red)
                valid_evi = evi_denom > 0
                evi[valid_evi] = 2.5 * (nir[valid_evi] - red[valid_evi]) / evi_denom[valid_evi]
                indices["evi"] = evi
                
                # Calculate NDWI (Normalized Difference Water Index)
                # Can help identify ancient canals, water management systems
                ndwi = np.zeros_like(green)
                valid_mask = (green + nir) > 0
                ndwi[valid_mask] = (green[valid_mask] - nir[valid_mask]) / (green[valid_mask] + nir[valid_mask])
                indices["ndwi"] = ndwi
                
                # Calculate MSAVI (Modified Soil Adjusted Vegetation Index)
                # Better soil/vegetation discrimination for archaeological features
                msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2
                indices["msavi"] = msavi
                
                # Calculate combined anomaly mask from all indices
                anomaly_mask = np.zeros_like(ndvi, dtype=bool)
                
                # Process each vegetation index for anomalies
                for index_name, index_data in indices.items():
                    # Calculate local anomalies (difference from neighborhood average)
                    # Use different smoothing parameters based on index
                    if index_name == "ndvi":
                        index_smooth = filters.gaussian(index_data, sigma=2)
                        index_trend = filters.gaussian(index_data, sigma=15)
                    elif index_name == "evi":
                        index_smooth = filters.gaussian(index_data, sigma=2.5)  
                        index_trend = filters.gaussian(index_data, sigma=20)
                    else:
                        index_smooth = filters.gaussian(index_data, sigma=1.5)
                        index_trend = filters.gaussian(index_data, sigma=10)
                        
                    index_anomaly = index_smooth - index_trend
                    
                    # Different threshold based on sensitivity level and index
                    # Use percentile thresholding for adaptive sensitivity
                    percentile_threshold = 100 - (adjusted_sensitivity * 20)  # Higher sensitivity = lower percentile
                    
                    # For EVI and MSAVI, use more sensitive thresholds
                    if index_name in ["evi", "msavi"]:
                        percentile_threshold -= 5
                    
                    # Cap percentile between reasonable bounds
                    percentile_threshold = max(70, min(percentile_threshold, 95))
                    
                    # Calculate threshold for anomaly detection
                    threshold = np.percentile(np.abs(index_anomaly), percentile_threshold)
                    
                    # Add to combined anomaly mask
                    anomaly_mask = anomaly_mask | (np.abs(index_anomaly) > threshold)
                
                # Clean up mask using morphological operations
                # For archaeological features, we want to keep smaller details
                anomaly_mask = morphology.remove_small_objects(anomaly_mask, min_size=9)  # Smaller for archaeological features
                anomaly_mask = morphology.remove_small_holes(anomaly_mask, area_threshold=9)
                
                # Apply morphological closing to connect nearby features (like geoglyphs)
                anomaly_mask = morphology.closing(anomaly_mask, morphology.disk(2))
                
                # Label connected regions
                labeled_mask, num_features = morphology.label(anomaly_mask, return_num=True)
                
                logger.info(f"Found {num_features} raw vegetation anomalies before size filtering")
                
                # Process regions
                features = []
                for region_id in range(1, num_features + 1):
                    # Extract region
                    region_mask = labeled_mask == region_id
                    
                    # Calculate area
                    area_px = np.sum(region_mask)
                    area_m2 = area_px * pixel_width_m * pixel_height_m
                    
                    # Adjust size filters based on sensitivity
                    size_factor = 1.0 - (sensitivity * 0.5)  # More sensitive = smaller allowed features
                    
                    # Filter by size
                    min_area_px = (min_size_px * min_size_px * 0.5) * size_factor 
                    max_area_px = (max_size_px * max_size_px * 2.0) * (2.0 - size_factor)
                    
                    # Skip if outside size limits, but log for debugging
                    if area_px < min_area_px:
                        continue
                    
                    if area_px > max_area_px:
                        continue
                    
                    # Calculate region properties
                    y_indices, x_indices = np.where(region_mask)
                    
                    # Skip if no points
                    if len(y_indices) == 0 or len(x_indices) == 0:
                        continue
                        
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    
                    # Calculate center coordinates
                    y_center = (y_min + y_max) / 2
                    x_center = (x_min + x_max) / 2
                    
                    # Safety check - ensure we're within bounds
                    if y_center < 0 or y_center >= bands.shape[1] or x_center < 0 or x_center >= bands.shape[2]:
                        continue
                    
                    # Convert to geographic coordinates
                    try:
                        lon, lat = rasterio.transform.xy(transform, y_center, x_center)
                    except Exception as e:
                        logger.error(f"Error converting coordinates: {e}")
                        continue
                    
                    # Calculate dimensions
                    width_px = x_max - x_min + 1
                    height_px = y_max - y_min + 1
                    width_m = width_px * pixel_width_m
                    height_m = height_px * pixel_height_m
                    
                    # Calculate shape compactness (0-1, 1 being most compact like circle)
                    perimeter = np.sum(morphology.binary_dilation(region_mask) & ~region_mask)
                    compactness = (4 * np.pi * area_px) / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # For archaeological features, check if it resembles known patterns
                    is_geometric = compactness > 0.6
                    
                    # Linear features detection (ancient roads, walls)
                    is_linear = False
                    if width_px > 0 and height_px > 0:
                        aspect_ratio = max(width_px, height_px) / min(width_px, height_px)
                        is_linear = aspect_ratio > 3.0
                    
                    # Determine most likely archaeological shape type
                    if is_geometric and compactness > 0.8:
                        shape_type = "circular"  # Circular earthworks, ring ditches
                    elif is_geometric:
                        shape_type = "geometric"  # Geometric earthworks, geoglyphs
                    elif is_linear:
                        shape_type = "linear"     # Roads, walls, canals
                    else:
                        shape_type = "organic"    # Other organic anomalies
                    
                    # Calculate index statistics for confidence measure
                    feature_stats = {}
                    confidence_factors = []
                    
                    for index_name, index_data in indices.items():
                        index_values = index_data[region_mask]
                        if len(index_values) > 0:
                            index_mean = float(np.mean(index_values))
                            index_std = float(np.std(index_values))
                            feature_stats[f"{index_name}_mean"] = index_mean
                            feature_stats[f"{index_name}_std"] = index_std
                            
                            # Calculate local anomaly strength
                            if index_name == "ndvi":
                                anomaly_values = indices["ndvi"][region_mask] - filters.gaussian(indices["ndvi"], sigma=10)[region_mask]
                                anomaly_strength = float(np.mean(np.abs(anomaly_values)))
                                feature_stats["ndvi_anomaly"] = anomaly_strength
                                confidence_factors.append(min(anomaly_strength * 5, 0.9))
                    
                    # Compactness contributes to confidence for geometric features
                    if is_geometric or is_linear:
                        confidence_factors.append(compactness)
                    
                    # Calculate overall confidence as average of factors
                    confidence = np.mean(confidence_factors) if confidence_factors else 0.5
                    confidence = min(confidence * (sensitivity + 0.5), 0.95)  # Scale by sensitivity
                    
                    # Create feature dictionary
                    feature = {
                        "type": "vegetation_anomaly",
                        "confidence": float(confidence),
                        "coordinates": {
                            "lon": float(lon),
                            "lat": float(lat)
                        },
                        "size": {
                            "width_m": float(width_m),
                            "height_m": float(height_m),
                            "area_m2": float(area_m2)
                        },
                        "vegetation": feature_stats,
                        "shape": shape_type,
                        "compactness": float(compactness),
                        "metadata": {
                            "source": "satellite",
                            "detection_method": "vegetation_anomaly",
                            "file": str(satellite_path.name)
                        }
                    }
                    
                    features.append(feature)
                
                logger.info(f"Detected {len(features)} potential vegetation anomalies")
                return features
            else:
                logger.warning(f"Insufficient bands for vegetation analysis (only {bands.shape[0]} bands)")
                return []
    
    except Exception as e:
        logger.error(f"Error detecting vegetation anomalies: {e}")
        return []

def merge_nearby_features(
    features: List[Dict],
    max_distance_m: float = 100.0
) -> List[Dict]:
    """
    Merge features that are close to each other, likely part of the same site.
    This reduces duplicate detections and helps identify larger sites.
    
    Args:
        features: List of detected features
        max_distance_m: Maximum distance in meters to consider features as part of the same site
        
    Returns:
        List of merged features
    """
    if not features:
        return []
    
    # Convert to GeoDataFrame for spatial operations
    # Create Point geometries from coordinates
    geometries = [
        Point(feature["coordinates"]["lon"], feature["coordinates"]["lat"]) 
        for feature in features
    ]
    
    # Create initial GeoDataFrame with WGS84 coordinates
    gdf = gpd.GeoDataFrame(features, geometry=geometries, crs="EPSG:4326")
    
    # Find an appropriate UTM zone for the area
    # Most of Amazon is in UTM zones 18-23, but we'll calculate it dynamically
    # Get the mean longitude to determine UTM zone
    mean_lon = gdf.geometry.x.mean()
    utm_zone = int(np.floor((mean_lon + 180) / 6) + 1)
    
    # Determine if we're in northern or southern hemisphere
    mean_lat = gdf.geometry.y.mean()
    epsg = 32600 + utm_zone if mean_lat >= 0 else 32700 + utm_zone
    
    # Project to the appropriate UTM zone for accurate distance measurements
    gdf_utm = gdf.to_crs(epsg=epsg)
    
    # Now buffer in projected coordinates (actual meters)
    gdf_utm["buffer"] = gdf_utm.geometry.buffer(max_distance_m)
    
    # Find overlapping buffers
    merged_features = []
    processed = set()
    
    for i, feature in gdf_utm.iterrows():
        if i in processed:
            continue
            
        # Find overlapping features
        overlaps = gdf_utm[gdf_utm["buffer"].intersects(feature["buffer"])].index.tolist()
        
        if len(overlaps) <= 1:
            # No overlaps, keep original feature
            merged_features.append(features[i])
        else:
            # Merge overlapping features
            overlapping_features = [features[j] for j in overlaps]
            
            # Calculate weighted centroid based on confidence
            total_confidence = sum(f["confidence"] for f in overlapping_features)
            if total_confidence > 0:
                weighted_lon = sum(f["coordinates"]["lon"] * f["confidence"] for f in overlapping_features) / total_confidence
                weighted_lat = sum(f["coordinates"]["lat"] * f["confidence"] for f in overlapping_features) / total_confidence
            else:
                # Fallback to simple average if all confidences are 0
                weighted_lon = sum(f["coordinates"]["lon"] for f in overlapping_features) / len(overlapping_features)
                weighted_lat = sum(f["coordinates"]["lat"] for f in overlapping_features) / len(overlapping_features)
            
            # Get the feature with highest confidence as the base
            base_feature = max(overlapping_features, key=lambda f: f["confidence"])
            
            # Create merged feature
            merged_feature = base_feature.copy()
            merged_feature["coordinates"]["lon"] = float(weighted_lon)
            merged_feature["coordinates"]["lat"] = float(weighted_lat)
            
            # Combine size information (use maximum values)
            merged_feature["size"]["width_m"] = float(max(f["size"]["width_m"] for f in overlapping_features))
            merged_feature["size"]["height_m"] = float(max(f["size"]["height_m"] for f in overlapping_features))
            merged_feature["size"]["area_m2"] = float(max(f["size"]["area_m2"] for f in overlapping_features))
            
            # Update confidence based on number of overlapping features
            # More overlapping detections increase confidence
            confidence_boost = min(0.1 * len(overlaps), 0.3)  # Max 0.3 boost
            merged_feature["confidence"] = min(merged_feature["confidence"] + confidence_boost, 0.99)
            
            # Add metadata about the merge
            merged_feature["metadata"]["merged_count"] = len(overlaps)
            merged_feature["metadata"]["merged_types"] = list(set(f["type"] for f in overlapping_features))
            
            merged_features.append(merged_feature)
        
        # Mark all overlapping features as processed
        processed.update(overlaps)
    
    logger.info(f"Merged {len(features)} features into {len(merged_features)} sites")
    return merged_features

def filter_by_confidence(
    features: List[Dict],
    min_confidence: float = 0.7,
    max_features: int = 200
) -> List[Dict]:
    """
    Filter features by confidence level and limit the total number.
    
    Args:
        features: List of detected features
        min_confidence: Minimum confidence level (0-1)
        max_features: Maximum number of features to return
        
    Returns:
        Filtered list of features
    """
    # Sort by confidence (highest first)
    sorted_features = sorted(features, key=lambda f: f["confidence"], reverse=True)
    
    # Filter by confidence
    filtered = [f for f in sorted_features if f["confidence"] >= min_confidence]
    
    # Limit number of features
    if max_features > 0 and len(filtered) > max_features:
        filtered = filtered[:max_features]
        logger.info(f"Limited to {max_features} highest confidence features")
    
    logger.info(f"Filtered {len(features)} features to {len(filtered)} with confidence >= {min_confidence}")
    return filtered

def save_features(
    features: List[Dict],
    output_path: Optional[Path] = None
) -> Path:
    """
    Save detected features to GeoJSON file.
    
    Args:
        features: List of detected features
        output_path: Path to save features (None for automatic)
        
    Returns:
        Path to saved file
    """
    if not features:
        return None
        
    # Generate output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DATA_DIR / f"detected_features_{timestamp}.geojson"
    
    # Convert to GeoJSON
    geometries = [
        Point(feature["coordinates"]["lon"], feature["coordinates"]["lat"]) 
        for feature in features
    ]
    
    # Create GeoDataFrame with correct CRS specified
    gdf = gpd.GeoDataFrame(features, geometry=geometries, crs="EPSG:4326")
    
    # Save to file - GeoJSON standard requires WGS84 coordinates (EPSG:4326)
    # No need to project for saving as GeoJSON
    gdf.to_file(output_path, driver="GeoJSON")
    logger.info(f"Saved {len(features)} features to {output_path}")
    
    return output_path 