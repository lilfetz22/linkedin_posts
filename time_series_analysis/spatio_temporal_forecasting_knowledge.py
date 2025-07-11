#
# --- Spatio-Temporal Forecasting in Practice: Building a Knowledge-Based Graph ---
#
# This script demonstrates how to construct an adjacency matrix for a graph
# based on a known, pre-defined physical network (e.g., a road network).
#

import pandas as pd
import numpy as np
import networkx as nx
import folium


def create_knowledge_based_adj_matrix(
    num_nodes: int, known_connections: list
) -> np.ndarray:
    """
    Creates an adjacency matrix from a predefined list of connections.

    Args:
        num_nodes (int): The total number of nodes in the graph.
        known_connections (list): A list of tuples, where each tuple (u, v)
                                  represents a known connection between nodes.

    Returns:
        np.ndarray: The resulting N x N adjacency matrix.
    """
    # Create a graph object
    G = nx.Graph()

    # Add nodes to ensure the matrix is the correct size, even if some nodes are isolated
    G.add_nodes_from(range(num_nodes))

    # Add the known connections (edges)
    G.add_edges_from(known_connections)

    # Convert the NetworkX graph to a NumPy adjacency matrix
    # The matrix will be symmetric because we used an undirected graph (nx.Graph)
    adjacency_matrix = nx.to_numpy_array(G, dtype=int)

    return adjacency_matrix


def visualize_graph_on_map(adjacency_matrix: np.ndarray, locations_df: pd.DataFrame):
    """
    Visualizes the knowledge-based graph on a geographic map using Folium.

    Args:
        adjacency_matrix (np.ndarray): The N x N adjacency matrix.
        locations_df (pd.DataFrame): DataFrame with lat/lon for node positions.
    """
    G = nx.from_numpy_array(adjacency_matrix)
    degrees = [val for (node, val) in G.degree()]

    map_center = [locations_df["latitude"].mean(), locations_df["longitude"].mean()]
    m = folium.Map(location=map_center, zoom_start=13, tiles="CartoDB positron")

    # Add edges (physical connections) to the map
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
                    color="#2E8B57",
                    weight=3,
                    opacity=0.8,
                    tooltip="Physical Road Connection",
                ).add_to(m)

    # Add nodes to the map
    for i, row in locations_df.iterrows():
        node_radius = 5 + degrees[i] * 3
        folium.CircleMarker(
            location=(row["latitude"], row["longitude"]),
            radius=node_radius,
            popup=f"Sensor {i}<br>Road Connections: {degrees[i]}",
            color="#006400",
            fill=True,
            fill_color="#3CB371",
            fill_opacity=0.9,
        ).add_to(m)

    m.save("knowledge_graph_map.html")
    print("Interactive map saved to knowledge_graph_map.html")


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
    num_sensors = len(locations_df)

    # Step 2: Define the known physical network (e.g., road connections)
    # This is our "domain knowledge". We are explicitly defining the graph structure.
    print("--- Defining known physical connections ---")
    known_connections = [
        # The dense downtown grid
        (0, 1),
        (0, 3),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        # The uptown sensors are on a more linear road
        (5, 6),
        (6, 7),
        # A "highway" connection from downtown to the south-west cluster
        (4, 8),
        # A local road in the south-west
        (8, 9),
        # An arterial road connecting the downtown and uptown clusters
        (3, 5),
    ]
    print(known_connections)
    print("-" * 30)

    # Step 3: Create the adjacency matrix from this knowledge
    adjacency_matrix = create_knowledge_based_adj_matrix(
        num_nodes=num_sensors, known_connections=known_connections
    )

    print(f"--- Generated Adjacency Matrix (shape: {adjacency_matrix.shape}) ---")
    print(adjacency_matrix)
    print("-" * 30)

    # Step 4: Visualize the resulting physical graph on a geographic map
    print("Generating graph visualization...")
    visualize_graph_on_map(adjacency_matrix, locations_df)
