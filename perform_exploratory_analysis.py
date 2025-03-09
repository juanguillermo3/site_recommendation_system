"""
title: Perform Exploratory Analysis
description: Merges the data sources for spatial analysis and produces exploratory plots
image_path: {stores_summary.html,plot_store_location_*}
"""

import os
from spatial_features import zip_features
from stores_placements import stores_gdf
from spatial_frame import spatial_frame
from data_manager import StoreLocationDataManager
from plotly_styles import (
    save_plot_as_html, centered_title, methodological_clarification,
    transparent_background
    )

# Load environment variables
import os
from dotenv import load_dotenv
load_dotenv()

# Load environment variables
SHOW_OFF_STATES = [state.strip() for state in os.getenv("SHOW_OFF_STATES").split(',')]
SHOW_OFF_STATES

# Instantiate the data manager
data_manager = StoreLocationDataManager(
    stores_df=stores_gdf,
    spatial_frame=spatial_frame,
    zip_features=zip_features,
    zip_id_col="zip_code",  # Adjust if a different column name is used
    state_column="NAME"  # Adjust if a different column name is used for states
)

# Confirm successful instantiation
print("StoreLocationDataManager successfully instantiated.")

# Apply decorators one by one, with their specific parameters
data_manager.plot_stores_summary_per_state = transparent_background()(
    data_manager.plot_stores_summary_per_state
)

data_manager.plot_stores_summary_per_state = methodological_clarification(
    clarification_text="This analysis is based on 2024 store data.", 
    #words_per_line=80
)(
    data_manager.plot_stores_summary_per_state
)

data_manager.plot_stores_summary_per_state = centered_title(
    title_text="Store Summary per State", 
    title_coords=(0.5, 0.9)  # Adjust position if needed
)(
    data_manager.plot_stores_summary_per_state
)

data_manager.plot_stores_summary_per_state = save_plot_as_html(
    filepath="stores_summary.html"
)(
    data_manager.plot_stores_summary_per_state
)

# Step 5: Call the decorated method
data_manager.plot_stores_summary_per_state()

for state in SHOW_OFF_STATES:
    filepath = f"plot_store_location_{state}.html"
    
    # Dynamically wrap the function for each state
    decorated_function = save_plot_as_html(filepath=filepath)(
        data_manager.plot_store_location
    )
    
    # Call the function to generate and save the map
    print(f"Generating and saving plot for {state}...")
    decorated_function(state)
