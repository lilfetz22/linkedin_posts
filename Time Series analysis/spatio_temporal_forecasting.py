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


# --- Main execution block ---
if __name__ == "__main__":
    # Step 1: Simulate sensor location data
    # Let's create a dummy dataset of 10 sensors scattered around a city area
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
    print("--- Input Sensor Locations ---")
    print(locations_df)
    print("-" * 30)

    # Step 2: Define the distance threshold for building the graph
    # Let's say we want to connect sensors that are within 2.5 kilometers of each other.
    DISTANCE_THRESHOLD_KM = 2.5
    print(f"Distance Threshold: {DISTANCE_THRESHOLD_KM} km\n")

    # Step 3: Create the adjacency matrix
    adjacency_matrix = create_distance_based_adj_matrix(
        locations_df, threshold=DISTANCE_THRESHOLD_KM
    )

    print(f"--- Generated Adjacency Matrix (shape: {adjacency_matrix.shape}) ---")
    print(adjacency_matrix)
    print("-" * 30)

    # Step 4: Visualize the resulting graph
    print("Generating graph visualization...")
    visualize_graph(adjacency_matrix, locations_df)
