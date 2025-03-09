"""
title: plot styles
description: Defines several decorators to distributes a custom styles across the plotly figures of a project,
             thus providing standardization and a cohesive style. Full suport on plotly. Partial support on Folium
             maps
"""

import plotly.graph_objects as go
import functools
import textwrap
import functools
import plotly.graph_objects as go
import folium


def transparent_background():
    """
    Decorator to set a fully transparent background for a Plotly figure.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            fig = func(*args, **kwargs)
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',  # Fully transparent background
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
            )
            return fig
        return wrapper
    return decorator

def methodological_clarification(clarification_text, words_per_line=100):
    """
    Decorator to add methodological clarification text to a Plotly figure.
    - Wraps text every `words_per_line` words by inserting line breaks.
    - Adjusts margins for better positioning.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            fig = func(*args, **kwargs)
            wrapped_text = "<br>".join(textwrap.wrap(clarification_text, width=words_per_line))
            fig.add_annotation(
                text=wrapped_text,
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0,  # Moved 5% closer (was -0.02)
                xanchor="center", yanchor="top",
                font=dict(size=12, color="black"),
                align="center"
            )
            fig.update_layout(margin=dict(l=15, r=15, t=55, b=50))  # Adjusted margins closer
            return fig
        return wrapper
    return decorator


def centered_title(title_text, title_coords="tightly_integrated"):
    """
    Decorator to add a centered title slightly above the plot.
    
    Parameters:
    - title_text (str): The text of the title.
    - title_coords (tuple or str): 
        - A tuple (x, y) to manually position the title.
        - If "tightly_integrated", uses the default (0.5, 0.85).
    
    Defaults:
    - If `title_coords="tightly_integrated"`, places the title at (x=0.5, y=0.85).
    - If a tuple (x, y) is provided, it is used directly.
    """
    # Handle the default case
    if title_coords == "tightly_integrated":
        x, y = 0.5, 0.85
    elif isinstance(title_coords, tuple) and len(title_coords) == 2:
        x, y = title_coords
    else:
        raise ValueError("title_coords must be either 'tightly_integrated' or a tuple (x, y).")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            fig = func(*args, **kwargs)

            # Add the centered annotation
            fig.add_annotation(
                text=title_text,
                showarrow=False,
                xref="paper", yref="paper",
                x=x, y=y,
                xanchor="center", yanchor="bottom",
                font=dict(size=16, color="black", family="Arial"),
                align="center"
            )

            # Preserve existing margins while ensuring a reasonable top margin
            existing_margins = fig.layout.margin.to_plotly_json() if hasattr(fig.layout, "margin") else {}
            fig.update_layout(
                margin={**existing_margins, "t": max(existing_margins.get("t", 0), 50)}
            )

            return fig
        return wrapper
    return decorator


def apply_typography():
    """
    Decorator to enforce a rigorous and minimalist font style in Plotly figures.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            fig = func(*args, **kwargs)
            fig.update_layout(
                font=dict(family="Lato, sans-serif", size=14, color="black"),
                title=dict(font=dict(size=18, family="Lato, sans-serif", color="black", weight="bold")),
                xaxis=dict(title=dict(font=dict(size=14, family="Lato, sans-serif", color="black"))),
                yaxis=dict(title=dict(font=dict(size=14, family="Lato, sans-serif", color="black"))),
            )
            return fig
        return wrapper
    return decorator


def save_plot_as_html(filepath="plot.html"):
    """
    Decorator to save a Plotly figure or Folium map as an HTML file.
    
    - Detects if the returned object is a Plotly `go.Figure` or a Folium `Map`.
    - Saves appropriately and prints which case was matched.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            fig = func(*args, **kwargs)  # Call the decorated function
            
            if isinstance(fig, go.Figure):
                print(f"[INFO] Matched: Plotly Figure. Saving to {filepath}.")
                fig.write_html(filepath)

            elif isinstance(fig, folium.Map):
                print(f"[INFO] Matched: Folium Map. Saving to {filepath}.")
                fig.save(filepath)

            else:
                print("[WARNING] No match found. Returning object as is.")

            return fig  # Return the original figure/map
        return wrapper
    return decorator
