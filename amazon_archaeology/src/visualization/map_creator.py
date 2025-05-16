"""
Map and visualization tools for archaeological features.
Creates interactive maps and statistical plots for discovered sites.
"""

import os
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
from datetime import datetime

import folium
from folium.plugins import HeatMap, MarkerCluster
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point

from ..config import DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_feature_map(
    features: List[Dict],
    output_path: Optional[Path] = None,
    center: Optional[Tuple[float, float]] = None,
    include_heatmap: bool = False
) -> Path:
    """
    Create an interactive map visualization of detected features.
    Uses Folium (Leaflet.js) for cost-effective mapping.
    
    Args:
        features: List of detected features
        output_path: Path to save the map HTML (None for automatic)
        center: Map center coordinates (lon, lat), None to auto-center
        include_heatmap: Whether to include heatmap layer
        
    Returns:
        Path to saved HTML file
    """
    try:
        # Calculate center if not provided
        if center is None and features:
            lats = [f["coordinates"]["lat"] for f in features]
            lons = [f["coordinates"]["lon"] for f in features]
            center = (np.mean(lats), np.mean(lons))
        elif center is None:
            # Default to Amazon basin if no features and no center specified
            center = (-3.4653, -62.2159)
        else:
            # Reorder if needed (center should be lat, lon for Folium)
            center = (center[1], center[0])
        
        # Create map
        m = folium.Map(
            location=center,
            zoom_start=12,
            tiles="OpenStreetMap"
        )
        
        # Add additional base layers
        folium.TileLayer('Stamen Terrain').add_to(m)
        folium.TileLayer('CartoDB positron').add_to(m)
        
        # Create feature groups for different types
        groups = {}
        all_features = folium.FeatureGroup(name="All Features")
        
        # Add marker for each feature
        for feature in features:
            # Get coordinates
            lat = feature["coordinates"]["lat"]
            lon = feature["coordinates"]["lon"]
            
            # Determine icon and color based on feature type and confidence
            if feature["type"] in ["rectangle", "square"]:
                icon = "square"
            elif feature["type"] == "circle":
                icon = "circle"
            elif feature["type"] == "line":
                icon = "minus"
            elif feature["type"] == "vegetation_anomaly":
                icon = "leaf"
            else:
                icon = "question"
            
            # Color based on confidence
            confidence = feature["confidence"]
            if confidence >= 0.7:
                color = "darkgreen"
            elif confidence >= 0.5:
                color = "orange"
            else:
                color = "gray"
            
            # Create popup content
            popup_content = f"""
            <h4>Archaeological Feature</h4>
            <b>Type:</b> {feature["type"]}<br>
            <b>Confidence:</b> {confidence:.2f}<br>
            <b>Size:</b> {feature["size"]["width_m"]:.1f}m × {feature["size"]["height_m"]:.1f}m<br>
            <b>Area:</b> {feature["size"]["area_m2"]:.1f} m²<br>
            <b>Coordinates:</b> {lat:.5f}, {lon:.5f}<br>
            """
            
            if "elevation" in feature:
                popup_content += f"""
                <b>Elevation:</b> Mean {feature["elevation"]["mean"]:.1f}m, 
                Range {feature["elevation"]["range"]:.1f}m<br>
                """
            
            if "vegetation" in feature:
                popup_content += f"""
                <b>NDVI:</b> {feature["vegetation"]["ndvi_mean"]:.2f}<br>
                <b>Anomaly Strength:</b> {feature["vegetation"]["anomaly_strength"]:.2f}<br>
                """
            
            # Add source information
            popup_content += f"""
            <br><b>Detection Method:</b> {feature["metadata"]["detection_method"]}<br>
            <b>Source:</b> {feature["metadata"]["source"]}<br>
            """
            
            # Create marker
            marker = folium.Marker(
                [lat, lon],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"{feature['type']} ({confidence:.2f})",
                icon=folium.Icon(color=color, icon=icon, prefix='fa')
            )
            
            # Add to appropriate group
            feature_type = feature["type"]
            if feature_type not in groups:
                groups[feature_type] = folium.FeatureGroup(name=f"{feature_type.replace('_', ' ').title()}")
            
            groups[feature_type].add_child(marker)
            all_features.add_child(marker.copy())
        
        # Add heatmap if requested
        if include_heatmap and features:
            # Prepare data for heatmap, with confidence as weight
            heat_data = [
                [
                    feature["coordinates"]["lat"],
                    feature["coordinates"]["lon"],
                    feature["confidence"]
                ]
                for feature in features
            ]
            
            # Create heatmap layer
            heatmap = HeatMap(
                heat_data,
                name="Confidence Heatmap",
                min_opacity=0.3,
                max_zoom=15,
                radius=15,
                blur=10
            )
            
            # Add heatmap to map
            heat_group = folium.FeatureGroup(name="Confidence Heatmap")
            heat_group.add_child(heatmap)
            m.add_child(heat_group)
        
        # Add all feature groups to map
        for group_name, group in groups.items():
            m.add_child(group)
        
        # Add all features group
        m.add_child(all_features)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add fullscreen control
        folium.plugins.Fullscreen().add_to(m)
        
        # Add measure tool
        folium.plugins.MeasureControl(
            position='topleft',
            primary_length_unit='meters',
            secondary_length_unit='kilometers',
            primary_area_unit='sqmeters',
            secondary_area_unit='sqkilometers'
        ).add_to(m)
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = DATA_DIR / "visualizations"
            output_dir.mkdir(exist_ok=True, parents=True)
            output_path = output_dir / f"archaeological_map_{timestamp}.html"
        
        # Save map
        m.save(str(output_path))
        logger.info(f"Saved interactive map to {output_path}")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error creating feature map: {e}")
        raise
    
def create_feature_plots(
    features: List[Dict],
    output_dir: Optional[Path] = None
) -> List[Path]:
    """
    Create statistical plots of feature characteristics.
    
    Args:
        features: List of detected features
        output_dir: Directory to save plots (None for automatic)
        
    Returns:
        List of paths to saved plot files
    """
    try:
        # Return empty list if no features
        if not features:
            logger.warning("No features to plot")
            return []
        
        # Generate output directory if not provided
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = DATA_DIR / "visualizations" / f"plots_{timestamp}"
            output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set plot style
        sns.set(style="whitegrid")
        
        output_paths = []
        
        # Plot 1: Feature types distribution
        plt.figure(figsize=(10, 6))
        feature_types = [f["type"] for f in features]
        type_counts = {t: feature_types.count(t) for t in set(feature_types)}
        
        # Sort by count
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        types = [t[0].replace("_", " ").title() for t in sorted_types]
        counts = [t[1] for t in sorted_types]
        
        # Create bar plot
        ax = sns.barplot(x=types, y=counts)
        plt.title("Distribution of Archaeological Feature Types")
        plt.xlabel("Feature Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Add count labels
        for i, count in enumerate(counts):
            ax.text(i, count + 0.1, str(count), ha="center")
        
        # Save plot
        plot_path = output_dir / "feature_types.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        output_paths.append(plot_path)
        
        # Plot 2: Confidence distribution
        plt.figure(figsize=(10, 6))
        confidences = [f["confidence"] for f in features]
        
        # Create histogram
        ax = sns.histplot(confidences, bins=10, kde=True)
        plt.title("Distribution of Feature Confidence Scores")
        plt.xlabel("Confidence Score")
        plt.ylabel("Count")
        plt.axvline(x=0.7, color='r', linestyle='--', label="High Confidence Threshold (0.7)")
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / "confidence_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        output_paths.append(plot_path)
        
        # Plot 3: Feature size distribution
        plt.figure(figsize=(10, 6))
        areas = [f["size"]["area_m2"] for f in features]
        
        # Create histogram
        ax = sns.histplot(areas, bins=15, kde=True)
        plt.title("Distribution of Feature Areas")
        plt.xlabel("Area (m²)")
        plt.ylabel("Count")
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / "size_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        output_paths.append(plot_path)
        
        # Plot 4: Confidence vs. Size scatter plot
        plt.figure(figsize=(10, 8))
        
        # Extract data
        areas = [f["size"]["area_m2"] for f in features]
        confidences = [f["confidence"] for f in features]
        types = [f["type"] for f in features]
        
        # Create unique color for each type
        unique_types = list(set(types))
        type_color_map = {t: i for i, t in enumerate(unique_types)}
        colors = [type_color_map[t] for t in types]
        
        # Create scatter plot
        scatter = plt.scatter(areas, confidences, c=colors, alpha=0.7, s=50, cmap="viridis")
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=plt.cm.viridis(type_color_map[t] / len(unique_types)), 
                      markersize=10, label=t.replace("_", " ").title())
            for t in unique_types
        ]
        plt.legend(handles=legend_elements, title="Feature Type")
        
        plt.title("Feature Confidence vs. Size")
        plt.xlabel("Area (m²)")
        plt.ylabel("Confidence Score")
        plt.axhline(y=0.7, color='r', linestyle='--', label="High Confidence Threshold")
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / "confidence_vs_size.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        output_paths.append(plot_path)
        
        logger.info(f"Saved {len(output_paths)} plots to {output_dir}")
        return output_paths
    
    except Exception as e:
        logger.error(f"Error creating feature plots: {e}")
        return []


def export_to_geojson(
    features: List[Dict],
    output_path: Optional[Path] = None
) -> Path:
    """
    Export features to a GeoJSON file for use in GIS software.
    
    Args:
        features: List of feature dictionaries
        output_path: Path to save the GeoJSON file (None for auto-generation)
        
    Returns:
        Path to the saved GeoJSON file
    """
    # Generate output path if not provided
    if output_path is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DATA_DIR / f"features_{timestamp}.geojson"
    
    # Convert features to GeoDataFrame
    geometries = [
        Point(feature["coordinates"]["lon"], feature["coordinates"]["lat"]) 
        for feature in features
    ]
    
    gdf = gpd.GeoDataFrame(features, geometry=geometries, crs="EPSG:4326")
    
    # Save to GeoJSON
    gdf.to_file(output_path, driver="GeoJSON")
    
    return output_path 