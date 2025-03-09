"""
title: Spatial Baseline
description: Elaborates SpatialBaseline. It provides a baseline ML model to predict store placements
             from underlyng data about the zip codes.
"""

import warnings
import geopandas as gpd
import pandas as pd
import folium
import numpy as np
import plotly.express as px
from shapely.geometry import Point
from folium.plugins import Draw, MeasureControl
from branca.colormap import linear
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from spatial_correlation import SpatialCorrelation

class SpatialBaseline(SpatialCorrelation):
    
    #
    # (0) Initialization
    #
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Pass additional parameters to the parent class
        
        self.baseline_probabilities=None
        self.fitted_model=None
        self.fit_baseline()
        
    #
    # (1) Fit model baseline
    #
    def fit_baseline(self, neg_to_pos_ratio=5):
        
        # Retrieve the data from get_samples
        X_train, y_train, X_test, y_test = self.get_samples(neg_to_pos_ratio)

        # Train the model
        self.fitted_model = LogisticRegression()
        self.fitted_model.fit(X_train, y_train)

        # Evaluate the model
        predictions = self.fitted_model.predict(X_test)
        self.baseline_probabilities = self.fitted_model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, predictions)
        f1_score_value = f1_score(y_test, predictions)

        # Optionally print or return the metrics
        print("Accuracy:", accuracy)
        print("F1 Score:", f1_score_value)
        
    #
    # (2) plot probabilities
    #   
    
    def _compute_probabilities(self, state_frame):
        """
        Computes probabilities using the fitted model and merges them into the state frame.
        """

        if self.fitted_model is None:
            raise ValueError("The model has not been fitted yet. Fit the model before plotting probabilities.")

        # Merge spatial data with features
        merged_data = state_frame.merge(self.zip_features, how='left', on=self.zip_id_col)

        # Ensure all necessary feature columns exist
        feature_columns = self.zip_features.columns.tolist()
        valid_data = merged_data.dropna(subset=feature_columns)

        # Convert to numerical format
        X = valid_data[feature_columns].values.astype(np.float32)

        # Predict probabilities
        valid_data["probabilities"] = self.fitted_model.predict_proba(X)[:, 1]

        # Ensure alignment of IDs
        valid_data[self.zip_id_col] = merged_data[self.zip_id_col][valid_data.index]

        # Merge back into the original state frame
        probs = valid_data[[self.zip_id_col, "probabilities"]]
        merged_data = merged_data.merge(probs, on=self.zip_id_col, how="left")

        return merged_data
    #
    def plot_probabilities(self, state_name, probabilities=None):
        """
        Plot predicted probabilities for a given state using a choropleth map and store locations.
        
        If probabilities are provided, they are appended directly to the state frame.
        Otherwise, the probabilities are computed using the fitted model, with missing values imputed using the median.
        """

        # Ensure the state exists in spatial data
        if state_name not in self.unique_states:
            raise ValueError(f"State '{state_name}' not found in the spatial data.")

        # Extract state-specific spatial frame
        state_frame = self.spatial_frame[self.spatial_frame[self.state_column] == state_name].copy()

        if probabilities is not None:
            # Validate shape before appending probabilities
            if len(probabilities) != len(state_frame):
                raise ValueError("Provided probabilities do not match the number of state entries.")
            state_frame["probabilities"] = probabilities

        else:
            # Compute probabilities using the fitted model
            state_frame = self._compute_probabilities(state_frame)

            # Impute missing probabilities with the median
            if state_frame["probabilities"].isnull().any():
                median_prob = state_frame["probabilities"].median()
                state_frame["probabilities"].fillna(median_prob, inplace=True)

        #
        # (1.1) base map
        #

        # Compute map center based on state's geographic centroid
        centroid_y = state_frame.geometry.centroid.y.mean()
        centroid_x = state_frame.geometry.centroid.x.mean()

        # Create Folium map centered on the state
        map_location = folium.Map(
            location=[centroid_y, centroid_x],
            zoom_start=6,
            min_zoom=6  # Restrict zooming out
        )

        # Plot probabilities as a choropleth layer
        folium.Choropleth(
            geo_data=state_frame.to_json(),
            data=state_frame,  # Since probabilities are now directly in state_frame
            columns=[self.zip_id_col, "probabilities"],
            key_on=f"feature.properties.{self.zip_id_col}",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="Probability of Store Presence"
        ).add_to(map_location)

        #
        # (1.2) Layer in the store locations
        #
        colormap = linear.YlOrRd_09.scale(0, 1)

        # Match stores based on ZIP code
        matched_stores = self.stores_df[self.stores_df[self.zip_id_col].isin(state_frame[self.zip_id_col])]

        for _, row in matched_stores.iterrows():
            # Fetch probability for the store's ZIP code
            prob = state_frame.loc[state_frame[self.zip_id_col] == row[self.zip_id_col], "probabilities"]

            if not prob.empty:
                probability = prob.values[0]  # Extract the probability value
            else:
                probability = 0  # Default to 0 if no probability is found

            store_name = row.get("store_name", f"Store {_}")

            # Add store marker with probability-based coloring
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=8,
                popup=f"{store_name} - {probability:.2f}",
                color="black",  # Border color
                fill=True,
                fill_color=colormap(probability),  # Fill color based on probability
                fill_opacity=1  # Solid color fill
            ).add_to(map_location)

        # Add colormap legend
        colormap.add_to(map_location)

        return map_location  # Return the Folium map for display
