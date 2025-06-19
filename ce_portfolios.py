import networkx as nx
import numpy as np
import time
import multiprocessing as mp
from multiprocessing import Manager

from portfolio import dominatesWithCost, generateFeasiblePortfolios
from performance import expected_traffic_volumes

def costEfficientPortfolios(G: nx.Graph, pairs: list[tuple[int, int]], paths: dict[tuple[int, int], list[np.ndarray]], traffic_volumes: dict[tuple[int, int], float], extreme_points: np.ndarray, node_reinforcements: list[tuple[int, float]], costs: list[float], budget: float, verbose=False) -> tuple[list[list[int]], dict[tuple[int, ...], list[float]]]:
    """
    Parameters:
        G (networkx.Graph): The graph in its original state with no portfolios applied and no disruptions.
        extremePoints (np.array): List of extreme points of the set of feasible weights.
        nodeReinforcements (dict[int, float]): A dictionary where the key is the index of the node which can be reinforced and the value is the resulting reliability.
        costs (list[float]): The costs of the reinforcement actions.

    Returns:    
        Q (list[tuple[int, ..., int]]): The set of cost-efficient portfolios.
    """
    
    r = len(node_reinforcements)

    start = time.time()
    Q_F, portfolio_costs = generateFeasiblePortfolios(r, costs, budget)
    end = time.time()
    print(f"It took {(end - start):.2f} seconds to compute the {len(Q_F)} feasible portfolios")
    
    Q = [[0 for _ in range(r)]]
    print(f"Q^0: {Q}")

    performance = expected_traffic_volumes(G, pairs, paths, traffic_volumes, extreme_points)
    performances: dict[tuple[int, ...], list[float]] = {tuple(Q[0]): performance}

    for l in range(r):
        start = time.time()
        print("---------------------------------")
        print(f"Iteration {l+1}/{r}: ")

        indexes = [k for k in range(r) if k != l]
        newQ = list(filter(lambda q1: not all(not (all(q1[k] == q2[k] for k in indexes)) or q1[l] != 1 for q2 in Q), Q_F)) # Step 5, newQ = Q^l

        print(f"Number of portfolios considered in Q_{l+1}: {len(newQ)}")

        for portfolio in newQ:
            G_q = G.copy()

            # This for loops applies 'portfolio' to 'G_q'
            for k, (node, prob) in enumerate(node_reinforcements):
                if portfolio[k] == 1:
                    G_q.nodes[node]['reliability'] = prob


            # Next we can calculate the expected performances of G_q
            performance = expected_traffic_volumes(G_q, pairs, paths, traffic_volumes, extreme_points)
            performances[tuple(portfolio)] = performance # Step 6

        # The approach from Joaquin's paper
        dominated = list(filter(lambda q1: any(dominatesWithCost(performances[tuple(q2)], performances[tuple(q1)], portfolio_costs[tuple(q2)], portfolio_costs[tuple(q1)]) for q2 in Q), newQ))
        newQ = [q for q in newQ if q not in dominated]

        dominated_previous = list(filter(lambda q1: any(dominatesWithCost(performances[tuple(q2)], performances[tuple(q1)], portfolio_costs[tuple(q2)], portfolio_costs[tuple(q1)]) for q2 in newQ), Q))
        Q = [q for q in Q if q not in dominated_previous]

        Q.extend(newQ)

        print(f"Number of portfolios in Q^{l+1}: {len(Q)}")
        if verbose:
            print(f"Q^{l+1}:")
            for portfolio in Q:
                print(portfolio)

        end = time.time()
        print(f"Time for iteration {l+1}: {(end - start):.2f} seconds.")

    return Q, performances

def applyPortfolioAndComputePerformance(G: nx.Graph, 
                                        pairs: list[tuple[int, int]], 
                                        paths: dict[tuple[int, int], list[np.ndarray]], 
                                        traffic_volumes: dict[tuple[int, int], float], 
                                        extreme_points: np.ndarray, 
                                        node_reinforcements: list[tuple[int, float]], 
                                        portfolio: list[int], 
                                        performances_shared: dict[tuple[int, ...], list[float]]
                                        ):
    G_q = G.copy()
    for k, (node, prob) in enumerate(node_reinforcements):
        if portfolio[k] == 1:
            G_q.nodes[node]['reliability'] = prob
    performance = expected_traffic_volumes(G_q, pairs, paths, traffic_volumes, extreme_points)
    performances_shared[tuple(portfolio)] = performance

def costEfficientPortfolios_parallel(G: nx.Graph, pairs: list[tuple[int, int]], paths: dict[tuple[int, int], list[np.ndarray]], traffic_volumes: dict[tuple[int, int], float], extreme_points: np.ndarray, node_reinforcements: list[tuple[int, float]], costs: list[float], budget: float, verbose=False) -> tuple[list[list[int]], dict[tuple[int, ...], list[float]]]:
    """
    Parameters:
        G (networkx.Graph): The graph in its original state with no portfolios applied and no disruptions.
        extreme_points (np.array): List of extreme points of the set of feasible weights.
        node_reinforcements (dict[int, float]): A dictionary where the key is the index of the node which can be reinforced and the value is the resulting reliability.
        costs (list[float]): The costs of the reinforcement actions.

    Returns:    
        Q (list[tuple[int, ..., int]]): The set of cost-efficient portfolios.
    """
    
    r = len(node_reinforcements)

    start = time.time()
    Q_F, portfolio_costs = generateFeasiblePortfolios(r, costs, budget)
    end = time.time()
    print(f"It took {(end - start):.2f} seconds to compute the {len(Q_F)} feasible portfolios")
    
    Q = [[0 for _ in range(r)]]

    performance = expected_traffic_volumes(G, pairs, paths, traffic_volumes, extreme_points)
    performances: dict[tuple[int, ...], list[float]] = {tuple(Q[0]): performance}

    threshold = min(15, r)
    # First few iterations are so fast that there is no need to parallelize them, too much overhead
    for l in range(threshold):
        start = time.time()
        print("---------------------------------")
        print(f"Iteration {l+1}/{r}: ")

        indexes = [k for k in range(r) if k != l]
        newQ = list(filter(lambda q1: not all(not (all(q1[k] == q2[k] for k in indexes)) or q1[l] != 1 for q2 in Q), Q_F)) # Step 5, newQ = Q^l

        print(f"Number of portfolios considered in Q_{l+1}: {len(newQ)}")

        for portfolio in newQ:
            G_q = G.copy()

            # This for loops applies 'portfolio' to 'G_q'
            for k, (node, prob) in enumerate(node_reinforcements):
                if portfolio[k] == 1:
                    G_q.nodes[node]['reliability'] = prob

            # Next we can calculate the expected performances of G_q
            performance = expected_traffic_volumes(G_q, pairs, paths, traffic_volumes, extreme_points)
            performances[tuple(portfolio)] = performance # Step 6

        # The approach from Joaquin's paper
        dominated = list(filter(lambda q1: any(dominatesWithCost(performances[tuple(q2)], performances[tuple(q1)], portfolio_costs[tuple(q2)], portfolio_costs[tuple(q1)]) for q2 in Q), newQ))
        newQ = [q for q in newQ if q not in dominated]

        dominated_previous = list(filter(lambda q1: any(dominatesWithCost(performances[tuple(q2)], performances[tuple(q1)], portfolio_costs[tuple(q2)], portfolio_costs[tuple(q1)]) for q2 in newQ), Q))
        Q = [q for q in Q if q not in dominated_previous]

        Q.extend(newQ)

        print(f"Number of portfolios in Q^{l+1}: {len(Q)}")
        if verbose:
            print(f"Q^{l+1}:")
            for portfolio in Q:
                print(portfolio)

        end = time.time()
        print(f"Time for iteration {l+1}: {(end - start):.2f} seconds.")

    # The iterations threshold, ..., r-1 are parallelized
    for l in range(threshold, r):
        start = time.time()
        print("---------------------------------")
        print(f"Iteration {l+1}/{r}: ")

        indexes = [k for k in range(r) if k != l]
        newQ = list(filter(lambda q1: not all(not (all(q1[k] == q2[k] for k in indexes)) or q1[l] != 1 for q2 in Q), Q_F)) # Step 5, newQ = Q^l

        print(f"Number of portfolios considered in Q_{l+1}: {len(newQ)}")

        manager = Manager()
        performances_shared = manager.dict()

        with mp.Pool(mp.cpu_count()) as pool:
            pool.starmap(
                applyPortfolioAndComputePerformance,
                [(G, pairs, paths, traffic_volumes, extreme_points, node_reinforcements, q, performances_shared) for q in newQ]
            )

        performances.update(dict(performances_shared))

        performances.update(dict(performances_shared))

        # The approach from Joaquin's paper
        dominated = list(filter(lambda q1: any(dominatesWithCost(performances[tuple(q2)], performances[tuple(q1)], portfolio_costs[tuple(q2)], portfolio_costs[tuple(q1)]) for q2 in Q), newQ))
        newQ = [q for q in newQ if q not in dominated]

        dominated_previous = list(filter(lambda q1: any(dominatesWithCost(performances[tuple(q2)], performances[tuple(q1)], portfolio_costs[tuple(q2)], portfolio_costs[tuple(q1)]) for q2 in newQ), Q))
        Q = [q for q in Q if q not in dominated_previous]

        Q.extend(newQ)
        
        print(f"Number of portfolios in Q^{l+1}: {len(Q)}")
        if verbose:
            print(f"Q^{l+1}:")
            for portfolio in Q:
                print(portfolio)

        end = time.time()
        print(f"Time for iteration {l+1}: {(end - start):.2f} seconds.")

    return Q, performances