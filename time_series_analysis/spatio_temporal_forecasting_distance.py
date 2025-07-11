#
# --- Spatio-Temporal Forecasting in Practice: Building a Distance-Based Graph ---
#
# This script demonstrates how to construct an adjacency matrix for a graph
# where nodes are connected if they are within a specific geographic distance.
#

# Step 0: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
import networkx as nx
import matplotlib.pyplot as plt
import folium


def create_distance_based_adj_matrix(
    locations_df: pd.DataFrame, threshold: float, add_self_loops: bool = False
):
    """
    Computes an adjacency matrix based on geographic distance.

    Args:
        locations_df (pd.DataFrame): DataFrame with 'latitude' and 'longitude' columns.
        threshold (float): The distance threshold in kilometers for connecting two nodes.
        add_self_loops (bool): Whether to include self-loops in the adjacency matrix.

    Returns:
        np.ndarray: The resulting N x N adjacency matrix.
    """
    # Extract latitude and longitude, and convert to radians for Haversine formula
    coords = np.radians(locations_df[["latitude", "longitude"]].values)

    # Calculate the pairwise Haversine distance matrix
    # The result from pairwise_distances is a fraction of the Earth's radius
    dist_matrix = pairwise_distances(coords, metric="haversine")

    # Convert distance to kilometers (Earth's radius is approx. 6371 km)
    dist_matrix_km = dist_matrix * 6371

    # Create the adjacency matrix based on the threshold
    # A[i, j] = 1 if distance(i, j) <= threshold, otherwise 0
    adjacency_matrix = (dist_matrix_km <= threshold).astype(int)

    # Remove self-loops (connections from a node to itself) if not desired
    if not add_self_loops:
        np.fill_diagonal(adjacency_matrix, 0)

    print(f"--- Distance Matrix (in km) ---\n{np.round(dist_matrix_km, 2)}\n")

    return adjacency_matrix


def visualize_graph(adjacency_matrix: np.ndarray, locations_df: pd.DataFrame):
    """
    Visualizes the graph using NetworkX and Matplotlib.

    Args:
        adjacency_matrix (np.ndarray): The N x N adjacency matrix.
        locations_df (pd.DataFrame): DataFrame with 'latitude' and 'longitude' for node positions.
    """
    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adjacency_matrix)

    # Create a dictionary for node positions using actual lat/lon
    pos = {i: (row["longitude"], row["latitude"]) for i, row in locations_df.iterrows()}

    plt.figure(figsize=(10, 10))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="skyblue",
        node_size=500,
        edge_color="gray",
        font_size=10,
        font_weight="bold",
    )

    plt.title("Graph of Sensors Connected by Geographic Proximity")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()


# --- Re-using the functions from the previous step ---
# def create_distance_based_adj_matrix(locations_df, threshold, add_self_loops=False):
#     coords = np.radians(locations_df[['latitude', 'longitude']].values)
#     dist_matrix = pairwise_distances(coords, metric='haversine')
#     dist_matrix_km = dist_matrix * 6371
#     adjacency_matrix = (dist_matrix_km <= threshold).astype(int)
#     if not add_self_loops:
#         np.fill_diagonal(adjacency_matrix, 0)
#     return adjacency_matrix

# --- Main execution block with new visualization ---
if __name__ == "__main__":
    # Step 1: Simulate sensor location data
    data = {
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
    locations_df = pd.DataFrame(data)

    # Step 2: Define the distance threshold
    DISTANCE_THRESHOLD_KM = 2.5

    # Step 3: Create the adjacency matrix
    adjacency_matrix = create_distance_based_adj_matrix(
        locations_df, DISTANCE_THRESHOLD_KM
    )

    # --- NEW VISUALIZATION LOGIC ---

    # Step 4: Create a NetworkX graph to calculate node degrees for sizing
    G = nx.from_numpy_array(adjacency_matrix)
    degrees = [val for (node, val) in G.degree()]

    # Step 5: Create the interactive map with Folium
    # Center the map on the average lat/lon of our sensors
    map_center = [locations_df["latitude"].mean(), locations_df["longitude"].mean()]
    m = folium.Map(location=map_center, zoom_start=13, tiles="CartoDB positron")

    # Add edges (connections) to the map as lines
    for i in range(adjacency_matrix.shape[0]):
        for j in range(
            i + 1, adjacency_matrix.shape[1]
        ):  # Avoid duplicates and self-loops
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
                    locations=[loc1, loc2], color="gray", weight=1.5, opacity=0.8
                ).add_to(m)

    # Add nodes to the map
    for i, row in locations_df.iterrows():
        # Improvement 3: Visualize the connection radius
        folium.Circle(
            location=(row["latitude"], row["longitude"]),
            radius=DISTANCE_THRESHOLD_KM * 1000,  # Radius in meters
            color="skyblue",
            fill=True,
            fill_opacity=0.1,
            weight=0,
        ).add_to(m)

        # Improvement 2: Vary node size by degree (connectivity)
        node_radius = 5 + degrees[i] * 4  # Base size + size based on # of connections

        # Add the actual node marker
        folium.CircleMarker(
            location=(row["latitude"], row["longitude"]),
            radius=node_radius,
            popup=f"Sensor {i}<br>Connections: {degrees[i]}",
            color="#0077B5",  # A nice LinkedIn blue
            fill=True,
            fill_color="#3DA9DE",
            fill_opacity=0.9,
        ).add_to(m)

    # Save the map to an HTML file
    m.save("spatio_temporal_graph_map.html")
    print("Interactive map saved to spatio_temporal_graph_map.html")
