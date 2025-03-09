"""
title: Data Manager
description: Handles ingestion merge and exploratory plottin of 3 data sources
             for spatial analysis.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import plotly.express as px
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go

class StoreLocationDataManager:
    
    #
    # (0) initialization
    #
    def __init__(self, 
                 stores_df: pd.DataFrame, 
                 spatial_frame: gpd.GeoDataFrame, 
                 zip_features: pd.DataFrame, 
                 zip_id_col: str = "zip_code",
                 state_column: str = "NAME"):
        #
        # inputs for spatial inference
        #
        self.stores_df = stores_df
        self.spatial_frame = spatial_frame
        self.zip_features = zip_features
        self.zip_id_col = zip_id_col
        self.state_column = state_column

        # Validate the common key and state column
        self._validate_zip_id_col()
        self._validate_state_column()

        # Precompute unique state names
        self.unique_states = set(self.spatial_frame[self.state_column].dropna().unique())
    #
    def _validate_zip_id_col(self):
        """Ensure the common key exists and is a string in all dataframes."""
        datasets = {
            "stores_df": self.stores_df,
            "spatial_frame": self.spatial_frame,
            "zip_features": self.zip_features
        }
        
        for dataset_name, dataset in datasets.items():
            if self.zip_id_col not in dataset.columns:
                raise ValueError(f"Common key '{self.zip_id_col}' not found in {dataset_name}.")
            if not pd.api.types.is_string_dtype(dataset[self.zip_id_col]):
                raise TypeError(f"Common key '{self.zip_id_col}' in {dataset_name} must be of type str.")
            if dataset[self.zip_id_col].isna().any():
                raise ValueError(f"Common key '{self.zip_id_col}' contains NaN values in {dataset_name}.")
    #
    def _validate_state_column(self):
        """Ensure the state column exists and contains valid data."""
        if self.state_column not in self.spatial_frame.columns:
            raise ValueError(f"State column '{self.state_column}' not found in spatial_frame.")
        if not pd.api.types.is_string_dtype(self.spatial_frame[self.state_column]):
            raise TypeError(f"State column '{self.state_column}' must be of type str.")
        if self.spatial_frame[self.state_column].isna().any():
            raise ValueError(f"State column '{self.state_column}' contains NaN values.")

    #
    # (1) Plotting store locations on a map
    #
    def plot_store_location(self, 
                            state_name: str, 
                            zoom_start: int = 6, 
                            min_zoom: int = 6
                            ):
        """Generate an interactive map showing store locations in the given state."""
        
        # Validate state name
        if state_name not in self.unique_states:
            raise ValueError(f"State '{state_name}' not found in the spatial data.")

        # Filter spatial data for the given state
        state_frame = self.spatial_frame[self.spatial_frame[self.state_column] == state_name]

        # Compute map center
        centroid = state_frame.geometry.centroid
        map_center = [centroid.y.mean(), centroid.x.mean()]
        
        # Create map
        map_location = folium.Map(
            location=map_center, 
            zoom_start=zoom_start,
            min_zoom=min_zoom
        )
        
        # Add state boundary
        folium.GeoJson(
            state_frame.to_json(),
            name=state_name,
            style_function=lambda feature: {
                'fillColor': 'blue',
                'color': 'white',
                'weight': 0.5,
                'fillOpacity': 0.4
            }
        ).add_to(map_location)

        
        # Find matching stores in the state
        matched_stores = self.stores_df[self.stores_df[self.zip_id_col].isin(state_frame[self.zip_id_col])]
        for _, row in matched_stores.iterrows():
            store_name = row.get('store_name', "Unnamed Store")
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=store_name
            ).add_to(map_location)
        
        return map_location
    
    #
    def _left_join(self, left_table: str, right_table: str) -> pd.DataFrame:
        """Perform a left join between two of the class's main tables using zip_id_col."""
        
        # Validate table names
        valid_tables = {
            "stores_df": self.stores_df,
            "spatial_frame": self.spatial_frame,
            "zip_features": self.zip_features
        }
        
        if left_table not in valid_tables:
            raise ValueError(f"Invalid left table '{left_table}'. Must be one of {list(valid_tables.keys())}.")
        if right_table not in valid_tables:
            raise ValueError(f"Invalid right table '{right_table}'. Must be one of {list(valid_tables.keys())}.")
        
        left_df = valid_tables[left_table]
        right_df = valid_tables[right_table]

        # Validate presence of zip_id_col
        if self.zip_id_col not in left_df.columns:
            raise ValueError(f"Key '{self.zip_id_col}' not found in {left_table}.")
        if self.zip_id_col not in right_df.columns:
            raise ValueError(f"Key '{self.zip_id_col}' not found in {right_table}.")
        
        return left_df.merge(right_df, on=self.zip_id_col, how='left')

    #
    def plot_stores_summary_per_state(self, household_col: str = "Hs", drop_highest_pct: float = 5.0):
        """Plots the number of stores vs household count per state, with a quadratic regression fit.
        Drops the top X% of states by household count before fitting.

        Args:
            household_col (str): Column name in zip_features containing household counts. Defaults to "Hs".
            drop_highest_pct (float): Percentage of states with the highest household counts to drop before regression. Defaults to 5%.
        """

        # (1) Get store count per state
        stores_per_state = self._left_join("stores_df", "spatial_frame")
        stores_count = stores_per_state[self.state_column].value_counts().reset_index()
        stores_count.columns = ["State", "Store Count"]

        # (2) Merge zip_features with spatial_frame to get households per zip
        zip_with_states = self._left_join("zip_features", "spatial_frame")

        # (3) Summarize household count per state
        if household_col not in zip_with_states.columns:
            raise ValueError(f"Column '{household_col}' is missing in zip_features.")
        households_per_state = zip_with_states.groupby(self.state_column)[household_col].sum().reset_index()
        households_per_state.columns = ["State", "Total Households"]

        # (4) Merge store count with household count
        merged_df = stores_count.merge(households_per_state, on="State", how="left")

        # (5) Remove top X% of states based on household count
        num_states_to_drop = int(len(merged_df) * (drop_highest_pct / 100))
        dropped_states = merged_df.nlargest(num_states_to_drop, "Total Households")["State"].tolist()
        filtered_df = merged_df[~merged_df["State"].isin(dropped_states)]

        # (6) Compute store density (Stores per 10,000 households)
        filtered_df["Stores per 10k Households"] = (filtered_df["Store Count"] / filtered_df["Total Households"]) * 10000

        # (7) Fit Quadratic Regression (2nd-degree polynomial)
        X = filtered_df["Total Households"]
        y = filtered_df["Store Count"]
        coeffs = np.polyfit(X, y, 2)  # Quadratic fit: y = ax² + bx + c
        poly_eq = np.poly1d(coeffs)
        X_fit = np.linspace(X.min(), X.max(), 100)
        y_fit = poly_eq(X_fit)

        # (8) Create Scatterplot
        fig = px.scatter(
            filtered_df,
            x="Total Households",
            y="Store Count",
            text="State",
            size="Stores per 10k Households",
            title="Number of Stores vs Household Count per State",
            labels={"Total Households": "Total Households", "Store Count": "Number of Stores"},
        )

        # (9) Add Regression Fit Line
        fig.add_trace(go.Scatter(
            x=X_fit,
            y=y_fit,
            mode="lines",
            line=dict(dash="dot", color="red", width=2),
            name="Quadratic Fit"
        ))

        # (10) Caption with dropped states
        dropped_caption = f"Dropped top {drop_highest_pct}% states (by household count): {', '.join(dropped_states)}"
        fig.add_annotation(
            text=dropped_caption,
            xref="paper", yref="paper",
            x=0.05, y=-0.15,
            showarrow=False,
            font=dict(size=12, color="gray"),
        )

        # (11) Style Updates
        fig.update_traces(marker=dict(color="blue", line=dict(width=2, color="white")), textposition="top center")
        fig.update_layout(title_x=0.5, xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))

        #fig.show()
        return fig
