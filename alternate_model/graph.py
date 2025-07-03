import networkx as nx
import random
import json
import numpy as np
import plotting

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
        G.add_node(i, pos=(pos_x, pos_y), label=i, reliability=random.uniform(0, 1))
    
    possible_edges = [(i, j) for i in range(num_nodes) for j in range(i+1, num_nodes)]
    
    selected_edges = random.sample(possible_edges, min(num_edges, len(possible_edges)))
    
    # Add these edges to the graph
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

    ids = {}
    i = 0
    for node in graph_data['nodes']:
        if node['type'] == 'primary':
            ids[node['id']] = i
            i += 1
        else:
            ids[node['id']] = node['id']

    for node in graph_data['nodes']:
        G.add_node(ids[node['id']], pos=(node['x'], node['y']), label=node['id'], type=node['type'], reliability=1.0)

    for edge in graph_data['edges']:
        n0 = ids[edge['n0']]
        n1 = ids[edge['n1']]
        G.add_edge(n0, n1)

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

def add_reliabilities(G: nx.Graph, reliabilities: dict[int, float]) -> nx.Graph:
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

def construct_graph(filename: str, reliabilities: dict[int, float]) -> nx.Graph:
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
    
    G = read_from_json(filename)
    G = remove_secondary_nodes(G)
    G = add_reliabilities(G, reliabilities)
    
    return G


if __name__ == '__main__':
    filename = 'data/network/sij.json'
    num_nodes = 40

    reliabilities = {i: 0.99 for i in range(num_nodes)}
    terminal_nodes = np.array([2, 11, 32]) # East, West, South
    for node in terminal_nodes:
        reliabilities[node] = 1.0
    
    G = construct_graph(filename, reliabilities)

    plotting.plot_network(G)

    
        