import networkx as nx
import numpy as np

from reliability import terminal_pair_reliability

# TODO: Perhaps parallelize this
def expected_traffic_volumes(G: nx.Graph,
                             paths: dict[tuple[int, int], list[np.ndarray]],
                             traffic_volumes: dict[tuple[int, int], float],
                             extreme_points: np.ndarray
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
    u_t: np.ndarray = np.zeros(extreme_points.shape[0])
    for i, (t, path) in enumerate(paths.items()):
        u_t[i] = traffic_volumes[t] * terminal_pair_reliability(G, path)

    U = np.matmul(extreme_points, u_t)
    
    return U