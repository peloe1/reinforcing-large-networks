import networkx as nx
import numpy as np

from reliability import terminal_pair_reliability

# Old deprecated function
def utility_functions(G: nx.Graph,
                      paths: dict[tuple[int, int], list[np.ndarray]]
                     ) -> list[float]:
    """
        Parameters: 
        -----------
        G: nx.Graph
            Graph object
        paths: dict[tuple[int, int], list[np.ndarray]]
            Dictionary mapping terminal node pairs to their corresponding feasible paths
        traffic_volumes: dict[tuple[int, int], float]
            Dictionary mapping terminal node pairs to traffic volumes
        extreme_points: np.ndarray
            List of extreme points

        Returns:
        --------
        U: list[float]
            List of expected traffic volumes for each extreme point
    """
    u_t: list[float] = [terminal_pair_reliability(G, path) for _, path in paths.items()]

    return u_t

def expected_travel(G: nx.Graph,
                    all_paths: dict[tuple[int, int], list[np.ndarray]],
                    travel_volumes: dict[tuple[int, int], float]
                    ) -> float:
    """
        Parameters: 
        -----------
        G: nx.Graph
            Graph object
        paths: dict[tuple[int, int], list[np.ndarray]]
            Dictionary mapping terminal node pairs to their corresponding feasible paths
        traffic_volumes: dict[tuple[int, int], float]
            Dictionary mapping terminal node pairs to traffic volumes
        extreme_points: np.ndarray
            List of extreme points

        Returns:
        --------
        U: list[float]
            List of expected traffic volumes for each extreme point
    """
    expectation: float = sum(travel_volumes[terminal_pair] * terminal_pair_reliability(G, path_set) for terminal_pair, path_set in all_paths.items())

    return expectation