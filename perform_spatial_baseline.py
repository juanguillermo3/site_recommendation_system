"""
title: Perform Spatial Baseline
description: Performs the training of baseline model to predict store placements from underlyng
             zip data
image_path: plot_probabilities_*
"""

from spatial_features import zip_features
from stores_placements import stores_gdf
from spatial_frame import spatial_frame
from spatial_baseline import SpatialBaseline
from plotly_styles import (
    save_plot_as_html
    )

# Load environment variables
import os
from dotenv import load_dotenv
load_dotenv()

# Load environment variables
HOLD_OUT_STATES = [state.strip() for state in os.getenv("HOLD_OUT_STATES").split(',')]
HOLD_OUT_STATES
# Inspect datasets
#zip_features.info()
#stores_df.info()
#spatial_frame.info()

# Instantiate the spatial baseline
sb =  SpatialBaseline(
    stores_df=stores_gdf,
    spatial_frame=spatial_frame,
    zip_features=zip_features,
    zip_id_col="zip_code",  # Adjust if a different column name is used
    state_column="NAME",  # Adjust if a different column name is used for states
    spatial_resolution=0.4
)

# Confirm successful instantiation
print("SpatialBaseline successfully instantiated.")

# Inspect datasets
for state in HOLD_OUT_STATES:
    filepath = f"plot_probabilities_{state}.html"
    
    # Wrap the function dynamically for each state
    decorated_function = save_plot_as_html(filepath=filepath)(
        sb.plot_probabilities
    )
    
    # Call the function to generate and save the plot
    print(f"Generating and saving probability plot for {state}...")
    decorated_function(state)
