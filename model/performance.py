import networkx as nx
import numpy as np

from reliability import terminal_pair_reliability

# Old deprecated function
def utility_functions(G: nx.Graph,
                      paths: dict[tuple[int, int], list[list[str]]]
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
                    all_paths: dict[tuple[str, str], list[list[str]]],
                    travel_volumes: dict[tuple[str, str], float]
                    ) -> tuple[float, dict[tuple[str, str], float]]:
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

    reliabilities: dict[tuple[str, str], float] = {pair: terminal_pair_reliability(G, path_set) for pair, path_set in all_paths.items()}

    expectation: float = sum(travel_volumes[terminal_pair] * reliabilities[terminal_pair] for terminal_pair, path_set in all_paths.items())

    return expectation, reliabilities

def expected_travel_hierarchical(Q: list[int],
                                 partitioned_paths: dict[tuple[str, str], tuple[list[tuple[str, str]], list[str]]], 
                                 terminal_pair_reliabilities: dict[str, dict[int, dict[tuple[str, str], float]]], 
                                 travel_volumes: dict[tuple[str, str], float]) -> float:
    
    neighbour_subnetworks: dict[str, str] = {'krm': 'kuo',
                                             'ohm': 'te',
                                             'lui': 'jki'
                                             }
    expectation: float = 0.0
    for terminal_pair, (path, sub_path) in partitioned_paths.items():
        reliability: float = 1.0
        for subnetwork, intermediate_pair, q in zip(sub_path, path, Q):
            if subnetwork not in ['ohm', 'krm', 'lui']:
                reliability_dict = terminal_pair_reliabilities[subnetwork][q]
                if intermediate_pair in reliability_dict:
                    reliability *= reliability_dict[intermediate_pair]
                else:
                    print(f"Looking at terminal pair {terminal_pair}")
                    print(f"The intermediate pair {intermediate_pair} is not in terminal pair reliabilities")
            else:
                reliability *= terminal_pair_reliabilities[neighbour_subnetworks[subnetwork]][q][intermediate_pair]
        
        expectation += reliability * travel_volumes[terminal_pair]

    return expectation