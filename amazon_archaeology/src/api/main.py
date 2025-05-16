"""
API server for the Amazon Archaeological Discovery Tool.
Provides RESTful endpoints for feature detection and visualization.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config import DATA_DIR, INITIAL_REGION_BOUNDS
from ..preprocessing.data_fetcher import get_data_sources_for_region, download_sample_lidar, fetch_sentinel_imagery
from ..analysis.feature_detector import detect_geometric_features, detect_vegetation_anomalies, merge_nearby_features, save_features
from ..utils.cache import clear_cache
from ..visualization.map_creator import create_feature_map, create_feature_plots

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Amazon Archaeological Discovery API",
    description="API for detecting potential archaeological sites in the Amazon basin",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class RegionBounds(BaseModel):
    """Region bounds model for API requests."""
    west: float = Field(..., description="Western longitude bound")
    south: float = Field(..., description="Southern latitude bound")
    east: float = Field(..., description="Eastern longitude bound")
    north: float = Field(..., description="Northern latitude bound")
    
    def to_tuple(self):
        """Convert to tuple format."""
        return (self.west, self.south, self.east, self.north)

class DetectionParams(BaseModel):
    """Parameters for feature detection."""
    min_size: int = Field(80, description="Minimum feature size in meters")
    max_size: int = Field(500, description="Maximum feature size in meters")
    merge_distance: float = Field(100.0, description="Distance in meters to merge nearby features")
    sensitivity: float = Field(0.5, description="Detection sensitivity (0-1)")

class DetectionRequest(BaseModel):
    """Request model for feature detection endpoint."""
    region: RegionBounds
    params: Optional[DetectionParams] = Field(None, description="Detection parameters")
    use_example_data: bool = Field(False, description="Use example data instead of fetching new data")
    create_visualization: bool = Field(False, description="Create visualization of results")

class VisualizationRequest(BaseModel):
    """Request model for visualization endpoint."""
    features: List[Dict]
    include_heatmap: bool = Field(True, description="Include heatmap in visualization")
    include_plots: bool = Field(True, description="Create statistical plots")

@app.get("/")
async def root():
    """Root endpoint providing API info."""
    return {
        "name": "Amazon Archaeological Discovery API",
        "version": "1.0.0",
        "description": "API for detecting potential archaeological sites in the Amazon basin",
        "documentation": "/docs",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "API info"},
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/sources", "method": "GET", "description": "Check available data sources for a region"},
            {"path": "/detect", "method": "POST", "description": "Detect archaeological features in a region"},
            {"path": "/visualize", "method": "POST", "description": "Create visualizations for detected features"},
            {"path": "/features", "method": "GET", "description": "Get previously detected features"},
            {"path": "/cache", "method": "DELETE", "description": "Clear cached data"}
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/sources")
async def get_sources(west: float, south: float, east: float, north: float):
    """
    Check available data sources for a specified region.
    
    Args:
        west: Western longitude bound
        south: Southern latitude bound
        east: Eastern longitude bound
        north: Northern latitude bound
        
    Returns:
        Dict of available data sources
    """
    try:
        bounds = (west, south, east, north)
        sources = get_data_sources_for_region(bounds)
        return sources
    except Exception as e:
        logger.error(f"Error getting data sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect")
async def detect_features(request: DetectionRequest, background_tasks: BackgroundTasks):
    """
    Detect archaeological features in a specified region.
    
    Args:
        request: Detection request parameters
        background_tasks: FastAPI background tasks
        
    Returns:
        Dict with detection results
    """
    try:
        # Extract parameters
        bounds = request.region.to_tuple()
        params = request.params or DetectionParams()
        
        # Get data paths
        lidar_path = None
        satellite_path = None
        
        if request.use_example_data:
            # Use example data
            logger.info("Using example data")
            example_dir = DATA_DIR / "example"
            lidar_path = example_dir / "lidar_sample.tif"
            satellite_path = example_dir / "satellite_sample.tif"
            
            # Check if example data exists, create synthetic data if not
            if not lidar_path.exists() or not satellite_path.exists():
                # This relies on the synthetic data generation logic in run.py
                # For API purposes, we'll just return an error if example data doesn't exist
                if not example_dir.exists():
                    example_dir.mkdir(exist_ok=True, parents=True)
                
                if not lidar_path.exists() or not satellite_path.exists():
                    return JSONResponse(
                        status_code=404,
                        content={"detail": "Example data not found. Please run with --use-example-data first to generate example data."}
                    )
        else:
            # Get available data sources
            sources = get_data_sources_for_region(bounds)
            
            # Download LiDAR data if available
            if sources["lidar"]["available"]:
                dataset_id = sources["lidar"]["sources"][0].get("datasetId", "default")
                lidar_path = download_sample_lidar(dataset_id, bounds)
            elif sources["srtm"]["available"]:
                lidar_path = download_sample_lidar("srtm", bounds)
            else:
                return JSONResponse(
                    status_code=404,
                    content={"detail": "No LiDAR or SRTM data available for this region"}
                )
            
            # Download Sentinel imagery if available
            if sources["sentinel"]["available"]:
                satellite_path = fetch_sentinel_imagery(bounds)
        
        # Detect features in LiDAR data
        lidar_features = detect_geometric_features(
            lidar_path,
            min_size=params.min_size,
            max_size=params.max_size
        )
        
        # Detect features in Sentinel imagery
        satellite_features = []
        if satellite_path:
            satellite_features = detect_vegetation_anomalies(
                satellite_path,
                min_size=params.min_size,
                max_size=params.max_size,
                sensitivity=params.sensitivity
            )
        
        # Combine all features
        all_features = lidar_features + satellite_features
        
        # Merge nearby features
        merged_features = merge_nearby_features(all_features, params.merge_distance)
        
        # Save features
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DATA_DIR / f"api_detected_features_{timestamp}.geojson"
        saved_path = save_features(merged_features, output_path)
        
        # Create visualizations if requested
        visualization_paths = []
        if request.create_visualization and merged_features:
            try:
                # Create in background to avoid blocking the API response
                def create_visualizations():
                    map_path = create_feature_map(
                        merged_features,
                        include_heatmap=True
                    )
                    plot_paths = create_feature_plots(merged_features)
                    logger.info(f"Created visualizations: {map_path}, {plot_paths}")
                
                background_tasks.add_task(create_visualizations)
                visualization_msg = "Visualizations are being created in the background"
            except Exception as e:
                logger.error(f"Error creating visualizations: {e}")
                visualization_msg = f"Error creating visualizations: {str(e)}"
        else:
            visualization_msg = "No visualizations requested"
        
        # Return results
        return {
            "timestamp": timestamp,
            "region": {
                "west": bounds[0],
                "south": bounds[1],
                "east": bounds[2],
                "north": bounds[3]
            },
            "parameters": {
                "min_size": params.min_size,
                "max_size": params.max_size,
                "merge_distance": params.merge_distance,
                "sensitivity": params.sensitivity
            },
            "counts": {
                "lidar_features": len(lidar_features),
                "satellite_features": len(satellite_features),
                "merged_features": len(merged_features),
                "high_confidence": len([f for f in merged_features if f["confidence"] >= 0.7])
            },
            "features": merged_features,
            "saved_path": str(saved_path) if saved_path else None,
            "visualization": visualization_msg
        }
    
    except Exception as e:
        logger.error(f"Error detecting features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visualize")
async def create_visualizations(request: VisualizationRequest):
    """
    Create visualizations for detected features.
    
    Args:
        request: Visualization request parameters
        
    Returns:
        Dict with visualization paths
    """
    try:
        features = request.features
        
        if not features:
            return JSONResponse(
                status_code=400,
                content={"detail": "No features provided"}
            )
        
        results = {}
        
        # Create interactive map
        if len(features) > 0:
            map_path = create_feature_map(
                features,
                include_heatmap=request.include_heatmap
            )
            results["map"] = str(map_path)
        
        # Create statistical plots
        if request.include_plots and len(features) > 0:
            plot_paths = create_feature_plots(features)
            results["plots"] = [str(p) for p in plot_paths]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "feature_count": len(features),
            "visualizations": results
        }
    
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features")
async def get_features(path: Optional[str] = None):
    """
    Get previously detected features.
    
    Args:
        path: Path to GeoJSON file (relative to data directory)
        
    Returns:
        Features from the specified file or a list of available feature files
    """
    try:
        if path:
            feature_path = DATA_DIR / path if not os.path.isabs(path) else Path(path)
            if not feature_path.exists():
                return JSONResponse(
                    status_code=404,
                    content={"detail": f"Feature file not found: {path}"}
                )
            
            with open(feature_path, "r") as f:
                features = json.load(f)
            
            return features
        else:
            # List available feature files
            geojson_files = list(DATA_DIR.glob("**/*.geojson"))
            return {
                "available_feature_files": [
                    {
                        "path": str(p.relative_to(DATA_DIR)),
                        "size_bytes": p.stat().st_size,
                        "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat()
                    }
                    for p in geojson_files
                ]
            }
    
    except Exception as e:
        logger.error(f"Error getting features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cache")
async def clear_api_cache():
    """
    Clear cached data to force fresh downloads and processing.
    
    Returns:
        Dict with count of cleared cache items
    """
    try:
        count = clear_cache()
        return {"cleared_items": count}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True) 