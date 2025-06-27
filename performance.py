import networkx as nx
import itertools
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

from reliability import terminal_pair_reliability
from graph import generate_random_graph_with_positions
from path import feasible_paths


# TODO: Perhaps parallelize this
def expected_traffic_volumes(G: nx.Graph, terminal_node_pairs: list[tuple[int, int]], paths: dict[tuple[int, int], list[np.ndarray]], traffic_volumes: dict[tuple[int, int], float], extreme_points: np.ndarray):
    """
        Parameters: 
        -----------
        G: nx.Graph
            Graph object
        terminal_node_pairs: list[tuple[int, int]]
            List of terminal node pairs
        paths: dict[tuple[int, int], list[np.ndarray]]
            Dictionary mapping terminal node pairs to feasible paths
        traffic_volumes: dict[tuple[int, int], float]
            Dictionary mapping terminal node pairs to traffic volumes
        extreme_points: np.ndarray
            List of extreme points

        Returns:
        --------
        U: list[float]
            List of expected traffic volumes for each extreme point
    """

    u_t: dict[tuple[int, int], float] = {}
    for t in terminal_node_pairs:
        u_t[t] = traffic_volumes[t] * terminal_pair_reliability(G, paths[t])

    U = []
    for point in extreme_points:
        U.append(sum([point[i] * u_t[t] for i, t in enumerate(terminal_node_pairs)]))
    
    return U