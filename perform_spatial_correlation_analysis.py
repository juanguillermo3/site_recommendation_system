"""
title: Perform Spatial Correlation Analysis
description: Performs the Spatial Correlation Analysis
image_path: {correlation_bars.html,plot_store_location_with_proximity_*}
"""

from spatial_features import zip_features
from stores_placements import stores_gdf
from spatial_frame import spatial_frame
from spatial_correlation import  SpatialCorrelation

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
# Inspect datasets
#zip_features.info()
#stores_df.info()
#spatial_frame.info()

# Instantiate the data manager
sc =  SpatialCorrelation(
    stores_df=stores_gdf,
    spatial_frame=spatial_frame,
    zip_features=zip_features,
    zip_id_col="zip_code",  # Adjust if a different column name is used
    state_column="NAME"  # Adjust if a different column name is used for states
)

# Confirm successful instantiation
print("SpatialCorrelation successfully instantiated.")

# Apply decorators one by one, with their specific parameters
plot_correlation_bars = transparent_background()(
    sc.plot_correlation_bars
)

plot_correlation_bars = methodological_clarification(
    clarification_text="This analysis is based on 2024 store data.", 
    #words_per_line=80
)(
    plot_correlation_bars
)

plot_correlation_bars = centered_title(
    title_text="Store Summary per State", 
    title_coords=(0.5, 0.9)  # Adjust position if needed
)(
    plot_correlation_bars
)

plot_correlation_bars = save_plot_as_html(
    filepath="correlation_bars.html"
)(
    plot_correlation_bars
)

# Step 5: Call the decorated method
plot_correlation_bars()

for state in SHOW_OFF_STATES:
    filepath = f"plot_store_location_with_proximity_{state}.html"
    
    # Wrap the function dynamically for each state
    decorated_function = save_plot_as_html(filepath=filepath)(
        sc.plot_store_location_with_proximity
    )
    
    # Call the function to generate and save the map
    print(f"Generating and saving proximity map for {state}...")
    decorated_function(state)
