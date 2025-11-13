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
                                 subnetworks: list[str],
                                 partitioned_paths: dict[tuple[str, str], tuple[list[tuple[str, str]], list[str]]], 
                                 terminal_pair_reliabilities: dict[str, dict[int, dict[tuple[str, str], float]]], 
                                 travel_volumes: dict[tuple[str, str], float]) -> float:
    
    subnetwork_map: dict[str, str] = {'krm': 'kuo',
                                      'ohm': 'te',
                                      'lui': 'jki',
                                      "apt": "apt", 
                                      "jki": "jki", 
                                      "knh": "knh", 
                                      "kuo": "kuo", 
                                      "lna": "lna", 
                                      "sij": "sij", 
                                      "skm": "skm", 
                                      "sor": "sor", 
                                      "te": "te", 
                                      "toi": "toi"
                                      }
    expectation: float = 0.0
    for terminal_pair, (path, sub_path) in partitioned_paths.items():
        reliability: float = 1.0

        if len(path) != len(sub_path):
            # This must be either KRM -> KUO, LUI -> JKI or OHM -> TE
            #print(f"The lengths of path {path} and sub_path {sub_path} don't match")
            intermediate_pair = path[0]
            subnetwork = sub_path[0]
            q = Q[subnetworks.index(subnetwork_map[subnetwork])]
            reliability *= terminal_pair_reliabilities[subnetwork_map[subnetwork]][q][intermediate_pair]
        
        else:
            for intermediate_pair, subnetwork in zip(path, sub_path):
                reliabilities_subnetwork = terminal_pair_reliabilities[subnetwork_map[subnetwork]]
                q = Q[subnetworks.index(subnetwork_map[subnetwork])]
                if q not in reliabilities_subnetwork:
                    print(f"Portfolio {q} is not in the dict of subnetwork {subnetwork}, which gets mapped to {subnetwork_map[subnetwork]}")
                elif intermediate_pair not in reliabilities_subnetwork[q]:
                    print(f"Intermediate pair {intermediate_pair} is not in the dict of subnetwork {subnetwork}, which gets mapped to {subnetwork_map[subnetwork]}")
                else:
                    reliability *= reliabilities_subnetwork[q][intermediate_pair]
        
        expectation += reliability * travel_volumes[terminal_pair]

    return expectation