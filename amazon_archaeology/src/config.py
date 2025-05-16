"""
Configuration module for the Amazon Archaeological Discovery Tool.
Loads settings from environment variables with cost-effective defaults.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))

# Ensure data directories exist
DATA_DIR.mkdir(exist_ok=True)
for subdir in ["lidar", "satellite", "historical", "cache"]:
    (DATA_DIR / subdir).mkdir(exist_ok=True)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_INITIAL = os.getenv("OPENAI_MODEL_INITIAL", "o3-mini")
OPENAI_MODEL_VALIDATION = os.getenv("OPENAI_MODEL_VALIDATION", "gpt-4")

# Google Earth Engine
GEE_SERVICE_ACCOUNT = os.getenv("GEE_SERVICE_ACCOUNT")
GEE_KEY_FILE = os.getenv("GEE_KEY_FILE", "/Users/geeknik/Dropbox/gee-key.json")

# Database Configuration (default to SQLite for cost-effectiveness)
DB_TYPE = os.getenv("DB_TYPE", "sqlite")
DB_PATH = os.getenv("DB_PATH", str(DATA_DIR / "spatial.db"))
DB_URL = ""

if DB_TYPE == "sqlite":
    DB_URL = f"sqlite:///{DB_PATH}"
elif DB_TYPE == "postgresql":
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "amazon_archaeology")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Data Paths
LIDAR_DATA_PATH = Path(os.getenv("LIDAR_DATA_PATH", DATA_DIR / "lidar"))
SATELLITE_DATA_PATH = Path(os.getenv("SATELLITE_DATA_PATH", DATA_DIR / "satellite"))
HISTORICAL_DATA_PATH = Path(os.getenv("HISTORICAL_DATA_PATH", DATA_DIR / "historical"))

# Processing Configuration
def parse_bounds(bounds_str: Optional[str] = None) -> Tuple[float, float, float, float]:
    """Parse region bounds from string in format 'long1,lat1,long2,lat2'"""
    if not bounds_str:
        # Default to a small region in the Amazon for cost-effective testing
        return (-73.0, -10.0, -72.0, -9.0)
    
    parts = bounds_str.split(",")
    if len(parts) != 4:
        raise ValueError("Bounds must be in format 'long1,lat1,long2,lat2'")
    
    return tuple(float(p) for p in parts)

INITIAL_REGION_BOUNDS = parse_bounds(os.getenv("INITIAL_REGION_BOUNDS"))
SAMPLE_RESOLUTION = int(os.getenv("SAMPLE_RESOLUTION", "30"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

# Cache Configuration (to minimize repeated API calls and data processing)
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
CACHE_DIR = Path(os.getenv("CACHE_DIR", DATA_DIR / "cache"))
CACHE_EXPIRE_DAYS = int(os.getenv("CACHE_EXPIRE_DAYS", "30"))

# OpenTopography API key
OPEN_TOPO_KEY = os.getenv("OPEN_TOPO_KEY", "demoapikeyot2020")  # Fallback to demo key if not set 