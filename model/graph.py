import networkx as nx
import random
import json
import numpy as np
#import plotting
import os
import plotly
from plotly.graph_objects import Figure

def generate_random_graph_with_positions(num_nodes, num_edges, pos_range=(0, 1)):
    """
        Parameters:
        -----------
        num_nodes: int
            Number of nodes in the graph
        num_edges: int
            Number of edges in the graph
        pos_range: tuple[float, float]
            Range of positions for the nodes

        Returns:
        --------
        G: nx.Graph
            A randomly generated Graph object
    
    """

    # Create an empty graph
    G = nx.Graph()
    
    # Add the specified number of nodes with random positions
    for i in range(num_nodes):
        # Generate random positions within the given range
        pos_x = random.uniform(pos_range[0], pos_range[1])
        pos_y = random.uniform(pos_range[0], pos_range[1])
        G.add_node(i, pos=(pos_x, pos_y), label=i, reliability=random.uniform(0.7, 0.95))  # Fixed reliability range
    
    # First ensure the graph is connected by creating a spanning tree
    nodes = list(G.nodes())
    random.shuffle(nodes)
    
    # Add edges to form a spanning tree
    for i in range(1, len(nodes)):
        G.add_edge(nodes[i-1], nodes[i], weight=1, color='gray')
    
    # Calculate remaining edges needed
    edges_added = len(nodes) - 1
    remaining_edges = num_edges - edges_added
    
    if remaining_edges > 0:
        # Get all possible edges that don't exist yet
        possible_edges = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if not G.has_edge(i, j):
                    possible_edges.append((i, j))
        
        # Add random edges from the remaining possible edges
        if possible_edges:
            selected_edges = random.sample(possible_edges, min(remaining_edges, len(possible_edges)))
            G.add_edges_from(selected_edges, weight=1, color='gray')
    
    return G


def read_from_json(filename: str) -> nx.Graph:
    """
        Parameters:
        -----------
        filename: str
            Path to the JSON file

        Returns:
        --------
        G: nx.Graph
            Graph object
    """

    with open(filename) as f:
        graph_data = json.load(f)
    
    G = nx.Graph()
            
    for node in graph_data['nodes']:
        G.add_node(node['id'], pos=(node['x'], node['y']), label=node['id'], type=node['type'], reliability=1.0)

    for edge in graph_data['edges']:
        G.add_edge(edge['n0'], edge['n1'])

    return G

# Removes nodes of type "secondary" while preserving the connections
# it connects each neighbour of a secondary node to the other neighbours of that secondary node
def remove_secondary_nodes(G):
    """
        Parameters:
        -----------
        G: nx.Graph
            Graph object

        Returns:
        --------
        G: nx.Graph
            The modified graph without any secondary nodes
    """

    secondary_nodes = [node for node, data in G.nodes(data=True) if data.get('type') == 'secondary']

    for node in secondary_nodes:
        neighbors = list(G.neighbors(node))
        
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                G.add_edge(neighbors[i], neighbors[j])

        G.remove_node(node)

    return G

def add_reliabilities(G: nx.Graph, reliabilities: dict[str, float]) -> nx.Graph:
    """
        Parameters:
        -----------
        G: nx.Graph
            Graph object
        probabilities: dict[int, float]
            Dictionary mapping node IDs to reliabilities

        Returns:
        --------
        G: nx.Graph
            The modified graph

    """

    for node, reliability in reliabilities.items():
        G.nodes[node]['reliability'] = reliability
    return G

def construct_graph(filename: str) -> tuple[nx.Graph, nx.Graph]:
    """
        Parameters:
        -----------
        filename: str
            Path to the JSON file
        terminal_nodes: list[int]
            List of terminal nodes
        travel_data: dict[tuple[int, int], float]
            Dictionary mapping terminal node pairs to travel frequencies
        probabilities: dict[int, float]
            Dictionary mapping node IDs to reliabilities

        Returns:
        --------
        G: nx.Graph
            Graph object
    
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = os.path.join(project_root, filename)

    G_original = read_from_json(filename)
    G = remove_secondary_nodes(G_original.copy())
    
    return G, G_original

def merge_graphs_from_json(file_list: list[str]) -> nx.Graph:
    """
        Parameters:
        -----------
        file_list: list[str]
            List of JSON file paths to merge (networks must share common nodes)

        Returns:
        --------
        G_merged: nx.Graph
            Unified graph with merged nodes and edges
    """

    G_merged = nx.Graph()

    for filename in file_list:
        G_temp = read_from_json(filename)

        # Merge nodes (preserving attributes)
        for node, attrs in G_temp.nodes(data=True):
            if not G_merged.has_node(node):
                G_merged.add_node(node, **attrs)
            else:
                # Update missing attributes only, do not overwrite existing
                for k, v in attrs.items():
                    if k not in G_merged.nodes[node]:
                        G_merged.nodes[node][k] = v

        # Merge edges (preserving weight + other attributes if present)
        for u, v, attrs in G_temp.edges(data=True):
            if not G_merged.has_edge(u, v):
                G_merged.add_edge(u, v, **attrs)

    return G_merged

def plot_network(G: nx.Graph):
    nodes = G.nodes(data=True)
    print(nodes)
    edges = G.edges()


    fig = Figure()

    xs = []
    ys = []

    for e in edges:
        xs.append(nodes[e[0]]["pos"][0])
        xs.append(nodes[e[1]]["pos"][0])
        xs.append(None)

        ys.append(nodes[e[0]]["pos"][1])
        ys.append(nodes[e[1]]["pos"][1])
        ys.append(None)

    fig = Figure()

    fig.add_scatter(name = "Track Geom.",
                    x = xs, 
                    y = ys,
                    line_color = "rgba(0, 0, 255, 0.3)")

    fig.add_scatter(name = "Track Infra", 
                    x = [data.get("pos")[0] for _, data in G.nodes(data=True) if data.get("type") == "primary"],
                    y = [data.get("pos")[1] for _, data in G.nodes(data=True) if data.get("type") == "primary"],
                    text = [node for node, data in G.nodes(data=True) if data.get("type") == "primary"],
                    mode = "markers")

    fig.update_layout(
                        xaxis_title = "<b>X Coord. (ETRS-TM35FIN)</b>",
                        yaxis_title = "<b>Y Coord. (ETRS-TM35FIN)</b>",
                        legend_orientation = "h",
                        height = 700,
                        yaxis_scaleanchor = "x", 
                        yaxis_scaleratio = 1,
                        )
    fig.show()



if __name__ == '__main__':
    
    filename = 'data/network/dipan_data/toi.json'#@network.json'

    G, G_original = construct_graph(filename)

    plot_network(G_original)

    reliabilities = {node: 0.99 for node in G.nodes()}
    G = add_reliabilities(G, reliabilities)

    for node, data in G.nodes(data=True):
        print(type(node))
        print("Node: ", node, " of type: ", data.get("type"), " has reliability: ", data.get("reliability"))

    #plotting.plot_network(G)

    
        