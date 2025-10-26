import numpy as np
import networkx as nx
import random

def terminal_pairs(terminal_nodes: list[int]) -> list[tuple[int, int]]:
    """
        Parameters:
        -----------
        terminal_nodes: list[int]
            A 1D NumPy array of terminal nodes.

        Returns:
        --------
        pairs: np.ndarray
            A 1D NumPy array of terminal node pairs represented by tuples of the form (u, v).
    
    """

    pairs: list[tuple[int, int]] = []
    for i, u in enumerate(terminal_nodes):
        for v in terminal_nodes[i+1:]:
            if u != v:
                pairs.append((u, v))
    
    return pairs

def simple_paths(G: nx.Graph, terminal_node_pairs: list[tuple[int, int]]) -> dict[tuple[int, int], list[np.ndarray]]:
    """
        Parameters:
        -----------
        G: nx.Graph
            Graph object
        terminal_node_pairs: np.ndarray
            A 1D NumPy array of terminal node pairs represented by tuples of the form (u, v).

        Returns:
        --------
        paths: np.ndarray
            A 2D NumPy array of simple paths between terminal node pairs.
    """

    paths: dict[tuple[int, int], list[np.ndarray]] = {}
    for u, v in terminal_node_pairs:
        paths[(u, v)] = list(nx.all_simple_paths(G, source=u, target=v))

    return paths

# TODO: This computes the turn angle based on the angle between the primary switches, 
# not based on the actual track geometry, perhaps store the angle in the node attributes when compiling the graph?
def filter_paths(G: nx.Graph, paths: list[np.ndarray]) -> list[np.ndarray]:
    """
        Parameters:
        -----------
        G: nx.Graph
            Graph object
        paths: np.ndarray
            A 2D NumPy array of simple paths

        Returns:
        --------
        feasible_paths: np.ndarray
            A 2D NumPy array of feasible paths (no turns greater than 90 degrees)
    """

    feasible_paths: list[np.ndarray] = []
    nodes = G.nodes(data=True)
    positions: dict[int, tuple[float, float]] = {node: data.get('pos') for node, data in nodes}
    for path in paths:
        feasible: bool = True
        for i in range(len(path) - 2):
            u, v, w = path[i], path[i+1], path[i+2]
            vec1: np.ndarray = np.array([positions[v][0] - positions[u][0], positions[v][1] - positions[u][1]])
            vec2: np.ndarray = np.array([positions[w][0] - positions[v][0], positions[w][1] - positions[v][1]])
            angle: float = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            
            # The path is not feasible if it includes even a single turn greater than 90 degrees
            if angle > np.pi / 2:
                feasible = False
                break
        if feasible:
            feasible_paths.append(path)

    #print(feasible_paths)

    return feasible_paths

def feasible_paths(G: nx.Graph, terminal_node_pairs: list[tuple[int, int]]) -> dict[tuple[int, int], list[np.ndarray]]:
    """
        Parameters:
        -----------
        G: nx.Graph
            Graph object
        terminal_nodes: np.ndarray
            A 1D NumPy array of terminal nodes

        Returns:
        --------
        path_dic: dict[tuple[int, int], np.ndarray]
            Dictionary mapping terminal node pairs to feasible paths
    """
    path_dic = {}
    for u, v in terminal_node_pairs:
        all_paths = nx.all_simple_paths(G, source=u, target=v)
        simple_paths: list[np.ndarray] = []
    
        for path in all_paths:
            simple_paths.append(np.array(path, dtype=int))
        
        feasible_paths = filter_paths(G, simple_paths)
        path_dic[(u, v)] = feasible_paths

    return path_dic


if __name__ == "__main__":
    num_nodes = 10
    num_edges = 20
    pos_range = [0, 1]

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

    terminal_nodes = [0, 3]
    terminal_node_pairs = terminal_pairs(terminal_nodes)

    for (u, v), paths in feasible_paths(G, terminal_node_pairs).items():
        print(type(paths[0]))