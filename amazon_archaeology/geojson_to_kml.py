#!/usr/bin/env python3
"""
Command-line utility to convert GeoJSON files to KML format for Google Earth.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.geo_converter import geojson_to_kml, convert_all_geojson_in_directory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the GeoJSON to KML converter."""
    parser = argparse.ArgumentParser(description="Convert GeoJSON files to KML for Google Earth")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Convert a single file
    file_parser = subparsers.add_parser("file", help="Convert a single GeoJSON file to KML")
    file_parser.add_argument(
        "input_file",
        help="Path to the GeoJSON file to convert"
    )
    file_parser.add_argument(
        "--output",
        help="Path to save the output KML file (default: same name with .kml extension)"
    )
    file_parser.add_argument(
        "--name",
        default="Amazon Archaeological Sites",
        help="Name for the KML document"
    )
    file_parser.add_argument(
        "--description",
        default="Potential archaeological sites detected by the Amazon Archaeological Discovery Tool",
        help="Description for the KML document"
    )
    
    # Convert all files in a directory
    dir_parser = subparsers.add_parser("directory", help="Convert all GeoJSON files in a directory")
    dir_parser.add_argument(
        "input_directory",
        help="Directory containing GeoJSON files to convert"
    )
    dir_parser.add_argument(
        "--output-directory",
        help="Directory to save the output KML files (default: same as input directory)"
    )
    dir_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for GeoJSON files in subdirectories"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if a command was specified
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Handle 'file' command
        if args.command == "file":
            if not os.path.exists(args.input_file):
                logger.error(f"Input file not found: {args.input_file}")
                return 1
                
            logger.info(f"Converting GeoJSON file: {args.input_file}")
            output_path = geojson_to_kml(
                args.input_file,
                args.output,
                args.name,
                args.description
            )
            
            logger.info(f"Successfully converted to KML: {output_path}")
            print(f"\n✅ KML file created: {output_path}")
            print("You can now open this file in Google Earth to visualize the archaeological sites.")
            return 0
            
        # Handle 'directory' command
        elif args.command == "directory":
            if not os.path.isdir(args.input_directory):
                logger.error(f"Input directory not found: {args.input_directory}")
                return 1
                
            logger.info(f"Converting GeoJSON files in directory: {args.input_directory}")
            output_files = convert_all_geojson_in_directory(
                args.input_directory,
                args.output_directory,
                args.recursive
            )
            
            if output_files:
                logger.info(f"Successfully converted {len(output_files)} files")
                print(f"\n✅ Converted {len(output_files)} GeoJSON files to KML:")
                for path in output_files:
                    print(f"  - {path}")
                print("\nYou can now open these files in Google Earth to visualize the archaeological sites.")
                return 0
            else:
                logger.warning("No GeoJSON files were converted")
                print("\n⚠️ No GeoJSON files were found or converted.")
                return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 