"""
title: Spatial Correlation
description: Elaborates StoreLocationDataManager. It provides methods to asses the correlation pattern
             between zip level features and the stores location.
"""

import os
import warnings
import geopandas as gpd
import pandas as pd
import folium
import plotly.express as px
from shapely.geometry import Point
from folium.plugins import Draw, MeasureControl
from sklearn.model_selection import train_test_split

from data_manager import StoreLocationDataManager

import os
from dotenv import load_dotenv
load_dotenv()

HOLD_OUT_STATES = [state.strip() for state in os.getenv("HOLD_OUT_STATES").split(',')]
HOLD_OUT_STATES

print(f"[INFO] Hold-out states: {HOLD_OUT_STATES}")

class SpatialCorrelation(StoreLocationDataManager):
    def __init__(self, spatial_resolution=0.2, hold_out_states=HOLD_OUT_STATES, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Pass additional parameters to the parent class
        self.spatial_resolution = spatial_resolution
        self.hold_out_states = hold_out_states
        self.determine_target_classification()  # Ensure raw_target is set at instantiation
    
    def determine_target_classification(self):
        """Classify locations as part of store proximity or holdout exclusions."""
        print(f"[INFO] Excluding states: {self.hold_out_states}")
        self.spatial_frame['raw_target'] = 0  # Initialize everything as neutral
        
        # Flag states to be excluded
        self.spatial_frame.loc[self.spatial_frame[self.state_column].isin(self.hold_out_states), 'raw_target'] = -1

        # Compute store buffers
        store_buffers = self.stores_df.geometry.buffer(self.spatial_resolution)

        # Find positive examples (store proximity), ensuring excluded states remain -1
        positives_geo = gpd.sjoin(
            self.spatial_frame[self.spatial_frame['raw_target'] == 0],  # Ignore excluded states
            gpd.GeoDataFrame(geometry=store_buffers), 
            how='inner', op='intersects'
        )

        self.spatial_frame.loc[positives_geo.index, 'raw_target'] = 1
    
    def get_samples(self, neg_to_pos_ratio=5, **kwargs):
        """Retrieve training and testing samples while respecting hold-out exclusions."""
        print(f"[INFO] Generating samples while excluding states: {self.hold_out_states}")
        neg_to_pos_ratio = int(neg_to_pos_ratio)

        self.spatial_frame['target'] = self.spatial_frame['raw_target']
        positives = self.spatial_frame[self.spatial_frame['target'] == 1]
        potential_negatives = self.spatial_frame[self.spatial_frame['target'] == 0]  # Use 0 instead of -1
        
        if len(positives) * neg_to_pos_ratio > len(potential_negatives):
            warnings.warn('Not enough negatives to match the ratio without replacement. Enabling replacement.')
            sampled_negatives = potential_negatives.sample(n=len(positives) * neg_to_pos_ratio, replace=True, random_state=42)
        else:
            sampled_negatives = potential_negatives.sample(n=len(positives) * neg_to_pos_ratio, replace=False, random_state=42)

        self.spatial_frame.loc[sampled_negatives.index, 'target'] = 0
        filtered_data = self.spatial_frame[self.spatial_frame['target'].isin([1, 0])]
        merged_data = filtered_data.merge(self.zip_features, how='left', on=self.zip_id_col)
        merged_data = merged_data.dropna()

        feature_columns = self.zip_features.columns.tolist()
        X = merged_data[feature_columns]
        y = merged_data['target']

        valid_split_params = {'test_size', 'train_size', 'random_state', 'shuffle', 'stratify'}
        split_params = {k: v for k, v in kwargs.items() if k in valid_split_params}

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.5, random_state=42, **split_params
        )

        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def plot_correlation_bars(self, k=20):
        X_train, y_train, _, _ = self.get_samples()
        data = pd.concat([X_train, y_train.rename('store_proximity')], axis=1)
        correlation_series = data.corr()['store_proximity'].drop('store_proximity')
        
        corr_df = correlation_series.abs().nlargest(k).reset_index()
        corr_df.columns = ['Feature', 'Correlation']
        
        fig = px.bar(
            corr_df, x='Feature', y='Correlation', text='Feature', color='Correlation',
            color_continuous_scale=px.colors.diverging.Tropic, title="Feature Correlation with Store Proximity"
        )
        
        fig.update_traces(
            marker=dict(line=dict(color='black', width=0.5), opacity=0.8),
            texttemplate='%{text}', textposition='inside',
            textfont=dict(color='white', size=12), insidetextanchor="start"
        )
        return fig
    
    def plot_store_location_with_proximity(self, state_name):
        if state_name not in self.spatial_frame[self.state_column].unique():
            raise ValueError(f"State '{state_name}' not found in the spatial data.")
        
        state_frame = self.spatial_frame[self.spatial_frame[self.state_column] == state_name]
        map_location = folium.Map(
            location=[state_frame.geometry.centroid.y.mean(), state_frame.geometry.centroid.x.mean()],
            zoom_start=6, min_zoom=6
        )
        
        Draw(export=True).add_to(map_location)
        MeasureControl().add_to(map_location)
        
        for target, color in zip([1, 0, -1], ['green', 'red', 'gray']):
            folium.GeoJson(
                state_frame[state_frame['target'] == target].to_json(),
                name=f"{'Positive' if target == 1 else 'Negative' if target == 0 else 'Neutral'} Examples",
                style_function=lambda x, color=color: {'fillColor': color, 'color': color, 'weight': 1, 'fillOpacity': 0.5}
            ).add_to(map_location)
        
        state_stores = self.stores_df[self.stores_df[self.zip_id_col].isin(state_frame[self.zip_id_col])]
        for _, row in state_stores.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']], radius=2.5, color='black', fill=True,
                fill_color='black', fill_opacity=0.5, popup=row.get('store_name', 'Store')
            ).add_to(map_location)
        
        return map_location
