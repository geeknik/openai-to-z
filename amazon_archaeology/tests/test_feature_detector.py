"""
Unit tests for the feature detection module.
Validates shape detection algorithms and error handling.
"""

import os
import sys
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock
import numpy as np
import rasterio
from rasterio.transform import from_origin

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.feature_detector import (
    detect_geometric_features,
    detect_vegetation_anomalies,
    merge_nearby_features,
    save_features
)

class TestFeatureDetector(unittest.TestCase):
    """Test cases for feature detection algorithms."""
    
    def setUp(self):
        """Set up test data."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_data_dir = Path(self.temp_dir.name)
        
        # Create test LiDAR data file
        self.lidar_path = self._create_test_lidar_data()
        
        # Create test satellite data file
        self.satellite_path = self._create_test_satellite_data()
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def _create_test_lidar_data(self):
        """Create synthetic LiDAR data for testing."""
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
        
        # Save to temporary file
        output_path = self.test_data_dir / "test_lidar.tif"
        with rasterio.open(
            output_path,
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
        
        return output_path
    
    def _create_test_satellite_data(self):
        """Create synthetic satellite data for testing."""
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
        
        # Create transform (each pixel is 10m)
        transform = from_origin(-64.0, -3.0, 0.001, 0.001)
        
        # Save to temporary file
        output_path = self.test_data_dir / "test_satellite.tif"
        with rasterio.open(
            output_path,
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
        
        return output_path
    
    def test_detect_geometric_features(self):
        """Test detection of geometric features in LiDAR data."""
        # Run feature detection
        features = detect_geometric_features(
            self.lidar_path,
            min_size=50,
            max_size=500,
            shapes_to_detect=["rectangle", "circle", "line"]
        )
        
        # Check that features were detected
        self.assertGreater(len(features), 0, "No geometric features detected")
        
        # Check that we have different shape types
        shape_types = set(feature["type"] for feature in features)
        self.assertGreaterEqual(len(shape_types), 2, "Expected at least 2 different shape types")
        
        # Check feature properties
        for feature in features:
            # Check coordinates
            self.assertIn("coordinates", feature)
            self.assertIn("lon", feature["coordinates"])
            self.assertIn("lat", feature["coordinates"])
            
            # Check size
            self.assertIn("size", feature)
            self.assertIn("width_m", feature["size"])
            self.assertIn("height_m", feature["size"])
            self.assertIn("area_m2", feature["size"])
            
            # Check elevation
            self.assertIn("elevation", feature)
            
            # Check confidence
            self.assertIn("confidence", feature)
            self.assertGreaterEqual(feature["confidence"], 0.0)
            self.assertLessEqual(feature["confidence"], 1.0)
            
            # Check metadata
            self.assertIn("metadata", feature)
            self.assertEqual(feature["metadata"]["source"], "lidar")
    
    def test_detect_vegetation_anomalies(self):
        """Test detection of vegetation anomalies in satellite data."""
        # Run vegetation anomaly detection
        features = detect_vegetation_anomalies(
            self.satellite_path,
            min_size=50,
            max_size=500,
            sensitivity=0.5
        )
        
        # Check that features were detected
        self.assertGreater(len(features), 0, "No vegetation anomalies detected")
        
        # Check feature properties
        for feature in features:
            # Check type
            self.assertEqual(feature["type"], "vegetation_anomaly")
            
            # Check coordinates
            self.assertIn("coordinates", feature)
            self.assertIn("lon", feature["coordinates"])
            self.assertIn("lat", feature["coordinates"])
            
            # Check size
            self.assertIn("size", feature)
            self.assertIn("width_m", feature["size"])
            self.assertIn("height_m", feature["size"])
            self.assertIn("area_m2", feature["size"])
            
            # Check vegetation properties
            self.assertIn("vegetation", feature)
            self.assertIn("ndvi_mean", feature["vegetation"])
            self.assertIn("anomaly_strength", feature["vegetation"])
            
            # Check confidence
            self.assertIn("confidence", feature)
            self.assertGreaterEqual(feature["confidence"], 0.0)
            self.assertLessEqual(feature["confidence"], 1.0)
            
            # Check metadata
            self.assertIn("metadata", feature)
            self.assertEqual(feature["metadata"]["source"], "satellite")
    
    def test_merge_nearby_features(self):
        """Test merging of nearby features."""
        # Create test features
        features = [
            {
                "type": "rectangle",
                "confidence": 0.8,
                "coordinates": {"lon": -64.0, "lat": -3.0},
                "size": {"width_m": 100, "height_m": 100, "area_m2": 10000},
                "metadata": {"source": "lidar", "detection_method": "geometric"}
            },
            {
                "type": "vegetation_anomaly",
                "confidence": 0.7,
                "coordinates": {"lon": -64.0001, "lat": -3.0001},  # Very close to first feature
                "size": {"width_m": 120, "height_m": 80, "area_m2": 9600},
                "metadata": {"source": "satellite", "detection_method": "vegetation_anomaly"}
            },
            {
                "type": "circle",
                "confidence": 0.6,
                "coordinates": {"lon": -64.1, "lat": -3.1},  # Far from other features
                "size": {"width_m": 50, "height_m": 50, "area_m2": 1963.5},
                "metadata": {"source": "lidar", "detection_method": "geometric"}
            }
        ]
        
        # Merge features
        merged_features = merge_nearby_features(features, max_distance_m=100.0)
        
        # Check result
        self.assertEqual(len(merged_features), 2, "Expected 2 merged features")
        
        # Check that confidence was boosted for merged feature
        merged_confidences = [f["confidence"] for f in merged_features]
        self.assertGreater(max(merged_confidences), 0.8, "Confidence should be boosted for merged features")
        
        # Check that merge metadata was added
        for feature in merged_features:
            if "merged_count" in feature["metadata"]:
                self.assertEqual(feature["metadata"]["merged_count"], 2, "Expected 2 features to be merged")
                self.assertIn("merged_types", feature["metadata"])
                self.assertEqual(len(feature["metadata"]["merged_types"]), 2, "Expected 2 merged types")
    
    def test_save_features(self):
        """Test saving features to GeoJSON."""
        # Create test features
        features = [
            {
                "type": "rectangle",
                "confidence": 0.8,
                "coordinates": {"lon": -64.0, "lat": -3.0},
                "size": {"width_m": 100, "height_m": 100, "area_m2": 10000},
                "metadata": {"source": "lidar", "detection_method": "geometric"}
            },
            {
                "type": "circle",
                "confidence": 0.6,
                "coordinates": {"lon": -64.1, "lat": -3.1},
                "size": {"width_m": 50, "height_m": 50, "area_m2": 1963.5},
                "metadata": {"source": "lidar", "detection_method": "geometric"}
            }
        ]
        
        # Save features
        output_path = self.test_data_dir / "test_features.geojson"
        saved_path = save_features(features, output_path)
        
        # Check that file was created
        self.assertTrue(saved_path.exists(), "GeoJSON file was not created")
        
        # Check that the path matches the requested path
        self.assertEqual(saved_path, output_path, "Returned path does not match requested path")
    
    def test_error_handling(self):
        """Test error handling in feature detection."""
        # Test with non-existent file
        features = detect_geometric_features(
            Path("non_existent_file.tif"),
            min_size=50,
            max_size=500
        )
        self.assertEqual(len(features), 0, "Expected empty list for non-existent file")
        
        # Test with invalid file
        invalid_file = self.test_data_dir / "invalid.tif"
        with open(invalid_file, "w") as f:
            f.write("This is not a valid GeoTIFF file")
        
        features = detect_geometric_features(
            invalid_file,
            min_size=50,
            max_size=500
        )
        self.assertEqual(len(features), 0, "Expected empty list for invalid file")
        
        # Test with empty feature list
        merged = merge_nearby_features([])
        self.assertEqual(len(merged), 0, "Expected empty list when merging empty list")
        
        # Test saving empty feature list
        result = save_features([])
        self.assertIsNone(result, "Expected None when saving empty feature list")

if __name__ == "__main__":
    unittest.main() 