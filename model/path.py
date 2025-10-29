import numpy as np
import networkx as nx
import random
from graph import *

def terminal_pairs(terminal_nodes: list[str]) -> list[tuple[str, str]]:
    """
        Parameters:
        -----------
        terminal_nodes: list[str]
            A 1D NumPy array of terminal nodes.

        Returns:
        --------
        pairs: np.ndarray
            A 1D NumPy array of terminal node pairs represented by tuples of the form (u, v).
    
    """

    pairs: list[tuple[str, str]] = []
    for i, u in enumerate(terminal_nodes):
        for v in terminal_nodes[i+1:]:
            if u != v:
                pairs.append((u, v))
    
    return pairs

# Old deprecated function
def simple_paths(G: nx.Graph, terminal_node_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], list[list[str]]]:
    """
        Parameters:
        -----------
        G: nx.Graph
            Graph object
        terminal_node_pairs: list[tuple[str, str]]
            A 1D NumPy array of terminal node pairs represented by tuples of the form (u, v).

        Returns:
        --------
        paths: dict[tuple[str, str], list[np.ndarray]]
            A 2D NumPy array of simple paths between terminal node pairs.
    """

    paths: dict[tuple[str, str], list[list[str]]] = {}
    for u, v in terminal_node_pairs:
        paths[(u, v)] = list(nx.all_simple_paths(G, source=u, target=v))

    return paths

# Old deprecated function
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
    positions: dict[str, tuple[float, float]] = {node: data.get('pos') for node, data in nodes}
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

# Old deprecated function
def old_feasible_paths(G: nx.Graph, terminal_node_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], list[np.ndarray]]:
    """
        Parameters:
        -----------
        G: nx.Graph
            Graph object
        terminal_nodes: np.ndarray
            A 1D NumPy array of terminal nodes

        Returns:
        --------
        path_dic: dict[tuple[str, str], np.ndarray]
            Dictionary mapping terminal node pairs to feasible paths
    """
    path_dic = {}
    for u, v in terminal_node_pairs:
        all_paths = nx.all_simple_paths(G, source=u, target=v)
        simple_paths: list[np.ndarray] = []
    
        for path in all_paths:
            simple_paths.append(np.array(path, dtype=str))
        
        feasible_paths = filter_paths(G, simple_paths)
        path_dic[(u, v)] = feasible_paths

    return path_dic

def feasible_paths(G_original: nx.Graph, 
                   G_simplified: nx.Graph, 
                   terminal_node_pairs: list[tuple[str, str]],
                   max_turn_angle: float = np.pi / 2) -> dict[tuple[str, str], list[list[str]]]:
    """
    Find feasible paths using original graph geometry but return simplified paths.
    
    Parameters:
    -----------
    G_original: nx.Graph
        Original graph with secondary nodes (for geometry validation)
    G_simplified: nx.Graph
        Simplified graph without secondary nodes (for path finding and return)
    terminal_node_pairs: list[tuple[str, str]]
        List of terminal node pairs
    max_turn_angle: float
        Maximum allowed turn angle in radians
    
    Returns:
    --------
    path_dic: dict[tuple[str, str], List[np.ndarray]]
        Dictionary mapping terminal node pairs to feasible paths in simplified graph
    """
    path_dic = {}
    
    for u, v in terminal_node_pairs:
        # Find all simple paths in the simplified graph (primary nodes only)
        all_paths_simplified = list(nx.all_simple_paths(G_simplified, source=u, target=v))
        feasible_paths_simplified = []
        
        for simplified_path in all_paths_simplified:
            # Reconstruct the detailed path in the original graph
            detailed_path = reconstruct_detailed_path(G_original, simplified_path)
            
            # Check if the detailed path has feasible turn angles
            if is_path_feasible(detailed_path, G_original, max_turn_angle):
                feasible_paths_simplified.append(simplified_path)
        
        path_dic[(u, v)] = feasible_paths_simplified
    
    return path_dic

def reconstruct_detailed_path(G_original: nx.Graph, simplified_path: list[str]) -> list[str]:
    """
    Reconstruct the detailed path including secondary nodes between primary nodes.
    
    Parameters:
    -----------
    G_original: nx.Graph
        Original graph with secondary nodes
    simplified_path: list[str]
        Path containing only primary nodes
    
    Returns:
    --------
    detailed_path: list[str]
        Complete path including secondary nodes
    """
    detailed_path = []
    
    for i in range(len(simplified_path) - 1):
        u, v = simplified_path[i], simplified_path[i+1]
        
        # Add the current primary node
        detailed_path.append(u)
        
        # Find the shortest path between primary nodes in the original graph
        # This will include the secondary nodes that define the actual track geometry
        try:
            sub_path = nx.shortest_path(G_original, u, v)
            # Add the intermediate nodes (excluding start and end to avoid duplicates)
            detailed_path.extend(sub_path[1:-1])
        except nx.NetworkXNoPath:
            # If no direct path exists, this shouldn't happen for valid simplified paths
            # Just continue with the primary nodes only
            continue
    
    # Add the final primary node
    detailed_path.append(simplified_path[-1])
    
    return detailed_path

def is_path_feasible(detailed_path: list[str], G: nx.Graph, max_turn_angle: float) -> bool:
    """
    Check if a detailed path has feasible turn angles.
    
    Parameters:
    -----------
    detailed_path: list[str]
        Path including secondary nodes
    G: nx.Graph
        Graph with position data
    max_turn_angle: float
        Maximum allowed turn angle in radians
    
    Returns:
    --------
    feasible: bool
        True if all turn angles are within limits
    """
    if len(detailed_path) < 3:
        return True  # No turns possible with less than 3 nodes
    
    positions = {node: G.nodes[node]['pos'] for node in detailed_path}
    
    for i in range(len(detailed_path) - 2):
        u, v, w = detailed_path[i], detailed_path[i+1], detailed_path[i+2]
        
        # Get positions for the three consecutive nodes
        pos_u = positions[u]
        pos_v = positions[v]
        pos_w = positions[w]
        
        if None in (pos_u, pos_v, pos_w):
            continue
            
        # Calculate vectors between consecutive nodes
        vec1 = np.array([pos_v[0] - pos_u[0], pos_v[1] - pos_u[1]])
        vec2 = np.array([pos_w[0] - pos_v[0], pos_w[1] - pos_v[1]])
        
        # Calculate the angle between vectors
        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norms < 1e-10:
            continue  # Avoid division by zero
            
        cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # Check if turn angle exceeds maximum
        if angle > max_turn_angle:
            return False
            
    return True

if __name__ == '__main__':
    filename = 'data/network/sij.json'
    num_nodes = 40
    
    # Get both graphs
    G, G_original = construct_graph(filename)
    reliabilities = {node: 0.99 for node in G.nodes()}
    G = add_reliabilities(G, reliabilities)

    terminal_nodes = ["SKM V0271", "APT V0001", "TOI V0002"]

    # Get terminal node pairs
    terminal_node_pairs = terminal_pairs(terminal_nodes)
    
    # Find feasible paths
    feasible_paths_dict = feasible_paths(G_original, G, terminal_node_pairs)

    # Validation of results
    #analyze_turn_constraints(G_original, G, terminal_node_pairs)
    #analyze_feasible_path_geometry(G_original, feasible_paths_dict)
    #check_simplification_quality(G_original, G)
    
    # Print results
    for (u, v), paths in feasible_paths_dict.items():
        print(f"Pair ({u}, {v}): {len(paths)} feasible paths")
        for path in paths:
            print(f"  Path: {path} \n")