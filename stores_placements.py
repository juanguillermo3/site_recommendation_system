"""
title: Store Placements Processing
description: This script processes store locations and standardizes ZIP codes.
"""

import os
import pandas as pd
import geopandas as gpd
from dotenv import load_dotenv
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load environment variables
load_dotenv()
APP_HOME = os.getenv("APP_HOME")

if not APP_HOME:
    raise ValueError("APP_HOME environment variable not found. Please set it in the .env file.")

# Change working directory
os.chdir(APP_HOME)

# Load the dataset
stores_df = pd.read_csv("data/trader_joes.csv")

# Standardizing ZIP code
stores_df["zip_code"] = stores_df["zip_code"].astype(str)

# Convert to GeoDataFrame
stores_gdf = gpd.GeoDataFrame(stores_df, geometry=gpd.points_from_xy(stores_df.longitude, stores_df.latitude))

# Display GeoDataFrame information
print(stores_gdf.info())

# Save the processed GeoDataFrame (optional)
# stores_gdf.to_file("data/processed_trader_joes.geojson", driver="GeoJSON")
# print("Processed GeoDataFrame saved.")
