"""
title: Spatial Frame Processing
description: This script processes spatial data from shapefiles and standardizes ZIP codes.
"""

import os
import geopandas as gpd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
APP_HOME = os.getenv("APP_HOME")

if not APP_HOME:
    raise ValueError("APP_HOME environment variable not found. Please set it in the .env file.")

# Change working directory
os.chdir(APP_HOME)

# Load the shapefile
spatial_frame = gpd.read_file("zips_with_states.shp")

# Standardizing ZIP code
spatial_frame["zip_code"] = spatial_frame["zip_code"].astype(str)

# Display DataFrame information
print(spatial_frame.info())

# Save the processed file (optional)
# spatial_frame.to_file("data/processed_zips_with_states.shp")
# print("Processed spatial data saved.")
