#
# --- Spatio-Temporal Forecasting in Practice: Building a Correlation-Based Graph ---
#
# This script demonstrates how to construct an adjacency matrix for a graph
# where nodes are connected if their time series are highly correlated.
#

import pandas as pd
import numpy as np
import folium
import networkx as nx


def simulate_correlated_time_series(
    num_sensors: int, num_timesteps: int
) -> pd.DataFrame:
    """
    Simulates time series data for a set of sensors with intentional correlations.

    Args:
        num_sensors (int): The number of sensors.
        num_timesteps (int): The number of time steps (e.g., hours).

    Returns:
        pd.DataFrame: A DataFrame of shape (num_timesteps, num_sensors) with sensor data.
    """
    print(f"--- Simulating {num_timesteps} timesteps for {num_sensors} sensors ---")
    time = np.arange(num_timesteps)

    # Create base patterns
    # Downtown pattern: Strong daily cycle + noise
    base_downtown_pattern = (
        np.sin(2 * np.pi * time / 24) * 20 + np.random.randn(num_timesteps) * 5
    )

    # Uptown pattern: Weaker daily cycle + different noise
    base_uptown_pattern = (
        np.sin(2 * np.pi * time / 24) * 10 + np.random.randn(num_timesteps) * 3
    )

    # Create the DataFrame
    ts_data = {}

    # Sensors 0-3: Tightly correlated "downtown" cluster
    for i in range(4):
        ts_data[f"S{i}"] = base_downtown_pattern + np.random.randn(num_timesteps) * 2

    # Sensor 4: Part of the downtown cluster but will be our "twin" for sensor 8
    ts_data["S4"] = base_downtown_pattern + np.random.randn(num_timesteps) * 2.5

    # Sensors 5-7: "Uptown" cluster
    for i in range(5, 8):
        ts_data[f"S{i}"] = base_uptown_pattern + np.random.randn(num_timesteps) * 1.5

    # Sensor 8: Geographically distant, but we'll make its behavior similar to S4
    ts_data["S8"] = (
        ts_data["S4"] * 1.1 + np.random.randn(num_timesteps) * 3
    )  # Functionally a "twin" of S4

    # Sensor 9: A lone wolf with a mostly random pattern
    ts_data["S9"] = np.random.randn(num_timesteps) * 10 + 15

    return pd.DataFrame(ts_data)


def create_correlation_based_adj_matrix(
    time_series_df: pd.DataFrame, threshold: float
) -> np.ndarray:
    """
    Computes an adjacency matrix based on time series correlation.

    Args:
        time_series_df (pd.DataFrame): DataFrame where columns are sensors and rows are time steps.
        threshold (float): The absolute correlation value above which to connect two nodes.

    Returns:
        np.ndarray: The resulting N x N adjacency matrix.
    """
    # Calculate the Pearson correlation matrix
    corr_matrix = time_series_df.corr()
    print(f"\n--- Correlation Matrix ---\n{np.round(corr_matrix, 2)}\n")

    # Create the adjacency matrix based on the absolute correlation threshold
    # A[i, j] = 1 if abs(corr(i, j)) >= threshold, otherwise 0
    adjacency_matrix = (np.abs(corr_matrix) >= threshold).astype(int)

    # Remove self-loops (the diagonal will always be 1)
    np.fill_diagonal(adjacency_matrix.values, 0)

    return adjacency_matrix.values


def visualize_graph_on_map(
    adjacency_matrix: np.ndarray, locations_df: pd.DataFrame, corr_threshold: float
):
    """
    Visualizes the functional graph on a geographic map using Folium.

    Args:
        adjacency_matrix (np.ndarray): The N x N adjacency matrix.
        locations_df (pd.DataFrame): DataFrame with lat/lon for node positions.
        corr_threshold (float): The correlation threshold used, for labeling.
    """
    G = nx.from_numpy_array(adjacency_matrix)
    degrees = [val for (node, val) in G.degree()]

    map_center = [locations_df["latitude"].mean(), locations_df["longitude"].mean()]
    m = folium.Map(location=map_center, zoom_start=13, tiles="CartoDB positron")

    # Add edges (connections) to the map
    for i in range(adjacency_matrix.shape[0]):
        for j in range(i + 1, adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] == 1:
                loc1 = (
                    locations_df.iloc[i]["latitude"],
                    locations_df.iloc[i]["longitude"],
                )
                loc2 = (
                    locations_df.iloc[j]["latitude"],
                    locations_df.iloc[j]["longitude"],
                )
                folium.PolyLine(
                    locations=[loc1, loc2],
                    color="#C70039",
                    weight=2.5,
                    opacity=0.8,
                    tooltip=f"Functional Connection (Corr >= {corr_threshold})",
                ).add_to(m)

    # Add nodes to the map
    for i, row in locations_df.iterrows():
        node_radius = 5 + degrees[i] * 3
        folium.CircleMarker(
            location=(row["latitude"], row["longitude"]),
            radius=node_radius,
            popup=f"Sensor {i}<br>Functional Connections: {degrees[i]}",
            color="#900C3F",
            fill=True,
            fill_color="#FF5733",
            fill_opacity=0.9,
        ).add_to(m)

    m.save("correlation_graph_map.html")
    print("Interactive map saved to correlation_graph_map.html")


# --- Main execution block ---
if __name__ == "__main__":
    # Step 1: Use the same sensor location data for consistency
    locations_data = {
        "sensor_id": [f"S{i}" for i in range(10)],
        "latitude": [
            34.0522,
            34.0531,
            34.0498,
            34.0600,
            34.0455,
            34.0722,
            34.0751,
            34.0688,
            34.0311,
            34.0250,
        ],
        "longitude": [
            -118.2437,
            -118.2510,
            -118.2399,
            -118.2458,
            -118.2555,
            -118.2800,
            -118.2851,
            -118.2750,
            -118.2650,
            -118.2700,
        ],
    }
    locations_df = pd.DataFrame(locations_data)

    # Step 2: Simulate correlated time series data for these sensors
    time_series_df = simulate_correlated_time_series(
        num_sensors=10, num_timesteps=720
    )  # 30 days of hourly data

    # Step 3: Define the correlation threshold
    # We want to connect nodes that are very strongly correlated (either positively or negatively)
    CORRELATION_THRESHOLD = 0.8
    print(f"Correlation Threshold: {CORRELATION_THRESHOLD}\n")

    # Step 4: Create the adjacency matrix based on correlation
    adjacency_matrix = create_correlation_based_adj_matrix(
        time_series_df, threshold=CORRELATION_THRESHOLD
    )

    print(f"--- Generated Adjacency Matrix (shape: {adjacency_matrix.shape}) ---")
    print(adjacency_matrix)
    print("-" * 30)

    # Step 5: Visualize the resulting functional graph on a geographic map
    print("Generating graph visualization...")
    visualize_graph_on_map(adjacency_matrix, locations_df, CORRELATION_THRESHOLD)
